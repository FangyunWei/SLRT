import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LogitsProcessor
from utils.misc import get_logger
from modelling.utils import PositionalEncoding, MaskedNorm, PositionwiseFeedForward, MLPHead
from modelling.S3D_base import SepConv3d
import math
import numpy as np


class SepConvVisualHead(torch.nn.Module):
    def __init__(self, cls_num, input_size=1024, hidden_size=512, ff_size=2048, pe=True,
                ff_kernelsize=3, pretrained_ckpt=None, is_empty=False, frozen=False, **kwargs):
        super().__init__()
        self.frozen = frozen
        self.input_size = input_size
        self.temp = kwargs.pop('temp', 1.0)
        if self.temp == 'learnable':
            self.temp = nn.Parameter(torch.tensor(0.0))
        self.topk = kwargs.pop('topk', 5)
        self.cnn_type = kwargs.pop('cnn_type', '3d')
        self.split_setting = kwargs.pop('split_setting', None)
        if self.split_setting is not None:
            if 'att' in self.split_setting:
                self.query_token = nn.Parameter(torch.randn(1,1,input_size))
                self.k_layer = nn.Linear(input_size, input_size)
                self.v_layer = nn.Linear(input_size, input_size)
                self.dropout = nn.Dropout(0.1)
            if 'ext_linear' in self.split_setting:
                self.fc = nn.Linear(input_size, cls_num)
            if 'bag_fc' in self.split_setting:
                self.bag_fc = nn.Linear(input_size, cls_num)
        
        self.gloss_output_layer = nn.Linear(input_size, cls_num)
        
        self.contras_setting = kwargs.pop('contras_setting', 'frame')
        if self.contras_setting is not None:
            self.word_emb_tab = kwargs.pop('word_emb_tab', None)
            self.word_emb_dim = kwargs.pop('word_emb_dim', 0)
            assert self.word_emb_tab is not None and self.word_emb_dim > 0
            self.word_emb_tab.requires_grad = False
            self.word_emb_sim = torch.matmul(F.normalize(self.word_emb_tab, dim=-1), F.normalize(self.word_emb_tab, dim=-1).T)
            self.word_emb_mapper = nn.Linear(self.word_emb_dim, input_size)
            self.vis_fea_mapper = nn.Identity()

            i_size = input_size*2 if 'concat' in self.contras_setting else input_size
            if 'dual' in self.contras_setting:
                self.word_fused_gloss_output_layer = nn.Linear(i_size, cls_num)
                if 'xmodal' in self.contras_setting:
                    self.xmodal_fused_gloss_output_layer = nn.Linear(i_size, cls_num)
                    self.xmodal_fused_gloss_output_layer.requires_grad_(False)

            if 'late' in self.contras_setting:
                if 'singlefc' not in self.contras_setting:
                    self.word_emb_mapper = nn.Sequential(nn.Linear(self.word_emb_dim, input_size//2),
                                                        nn.ReLU(inplace=True),
                                                        nn.Linear(input_size//2, input_size)
                                                        )
                if 'share' not in self.contras_setting:
                    self.word_fused_gloss_output_layer = nn.Linear(i_size, cls_num)
                if 'mlp' in self.contras_setting:
                    self.mlp = nn.Sequential(nn.Linear(i_size, i_size//4),
                                            nn.BatchNorm1d(i_size//4),
                                            nn.ReLU(True),
                                            nn.Linear(i_size//4, i_size),
                                            nn.BatchNorm1d(i_size),
                                            nn.ReLU(True))
                else:
                    self.mlp = nn.Identity()

            if '300d' in self.contras_setting:
                self.word_emb_mapper = nn.Identity()
                self.vis_fea_mapper = nn.Linear(input_size, self.word_emb_dim)
                if 'transformer' in self.contras_setting:
                    encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=8, dim_feedforward=2048, batch_first=True)
                    self.word_emb_ext = nn.TransformerEncoder(encoder_layer, num_layers=2)
                    self.word_emb_token = nn.Parameter(torch.randn(1,1,input_size))

        if pretrained_ckpt:
            self.load_from_pretrained_ckpt(pretrained_ckpt)
    
    
    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        logger = get_logger()
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {}
        for k,v in checkpoint.items():
            if 'recognition_network.visual_head.' in k:
                load_dict[k.replace('recognition_network.visual_head.','')] = v
        self.load_state_dict(load_dict, strict=True)
        logger.info('Load Visual Head from pretrained ckpt {}'.format(pretrained_ckpt))


    def forward(self, x, labels=None, fbank=None, temp_idx=None, bag_labels=None):
        if self.contras_setting is not None and 'xmodal' in self.contras_setting and fbank is not None:
            self.xmodal_fused_gloss_output_layer.requires_grad_(True)
        if x.ndim > 3 and self.cnn_type == '3d':
            x = F.avg_pool3d(x, (2, x.size(3), x.size(4)), stride=1)  #spatial global average pool
            x = x.view(x.size(0), x.size(1), x.size(2)).permute(0, 2, 1)  #B,T,C

        split_logits = []
        cam = k = logits = gloss_probabilities = word_fused_gloss_logits = word_fused_gloss_probabilities = topk_idx = None
        bag_logits = [None, None]
        if self.cnn_type != '2d' and x.ndim>2:
            if self.split_setting is not None and '4x' in self.split_setting:
                scale_factor = 8 if 'keepscale' in self.split_setting else 4
                x = F.interpolate(x.permute(0,2,1), scale_factor=scale_factor, mode='linear').permute(0,2,1)
                # print(self.split_setting)
            if self.split_setting is not None and 'split' in self.split_setting:
                if '4x' in self.split_setting:
                    up_fea = x
                else:
                    scale_factor = 4
                    if 'scale_8' in self.split_setting:
                        scale_factor = 8
                    elif 'scale_2' in self.split_setting:
                        scale_factor = 2
                    up_fea = F.interpolate(x.permute(0,2,1), scale_factor=scale_factor, mode='linear').permute(0,2,1)  #B,T,C
                
                if 'cam' in self.split_setting:
                    cam_w = self.gloss_output_layer.weight.index_select(0, labels).unsqueeze(1)  #B,1,C
                    cam = (up_fea*cam_w).sum(dim=-1)  #B,T
                    # cam = (cam - cam.amin(dim=-1, keepdim=True)) / (cam.amax(dim=-1, keepdim=True) - cam.amin(dim=-1, keepdim=True))

                elif 'att' in self.split_setting:
                    B, T, C = up_fea.shape
                    query = self.query_token.repeat(B,1,1)  #B,1,C
                    key = self.k_layer(up_fea)  #B,T,C
                    value = self.v_layer(up_fea)  #B,T,C
                    query = query / math.sqrt(self.input_size)
                    cam = torch.matmul(query, key.transpose(1,2))  #B,1,T
                    cam = F.softmax(cam, dim=-1)
                    up_fea = torch.matmul(self.dropout(cam), value).squeeze(1)
                    if '4x' in self.split_setting:
                        logits = self.gloss_output_layer(up_fea)
                    else:
                        logits = self.gloss_output_layer(x.mean(dim=1))
                        if 'ext_linear' in self.split_setting:
                            split_logits.append(self.fc(up_fea))
                        else:
                            split_logits.append(self.gloss_output_layer(up_fea))

                if 'nonblk' in self.split_setting and temp_idx is not None:
                    B, T, C = up_fea.shape
                    mask = torch.zeros(B,T).to(x.device)
                    for i in range(B):
                        mask[i, temp_idx[i,0]:temp_idx[i,1]] = 1
                        mask[i, temp_idx[i,1]:] = 2
                    mask = mask.unsqueeze(-1)  #B,T,1
                    m = (mask==1)
                    tot = m.sum(dim=1)  #B,1
                    tot = torch.clamp(tot, min=1)
                    fea = (m*up_fea).sum(dim=1) / tot  #B,C
                    if 'ext_linear' in self.split_setting:
                        s_logits = self.fc(fea)
                    else:
                        s_logits = self.gloss_output_layer(fea)
                    split_logits.append(s_logits)
                    # print(fea.shape)
            x = x.mean(dim=1)
            
        B, C = x.shape[0], x.shape[-1]
        vis_fea = x

        # interact with word_embs
        if logits is None:
            logits = self.gloss_output_layer(x)  #[B,N]

        if self.split_setting is not None and 'bag_fc' in self.split_setting and bag_labels is not None:
            bag_logits[0] = self.bag_fc(x)
            if 'both' in self.split_setting:
                bag_logits[1] = self.bag_fc(fea)
            # print(bag_logits[0].shape)

        if self.contras_setting is not None and 'dual' in self.contras_setting:
            if self.training:
                logits_data = logits.clone().detach()
                batch_idx = torch.arange(B)
                logits_data[batch_idx, labels] = float('-inf')
                idx = torch.argsort(logits_data, dim=1, descending=True)[:, :self.topk-1]  #[B,K-1]
                topk_idx = torch.cat([labels.unsqueeze(1), idx], dim=1)  #[B,K]
            else:
                topk_idx = torch.argsort(logits, dim=1, descending=True)[:, :5]
            topk_idx = topk_idx.reshape(-1)
            if k is None:
                k = self.word_emb_mapper(self.word_emb_tab)  #[N,C]
            word_embs = k.index_select(0, topk_idx).reshape(B,-1,C)  #[B,K,C]
            fused_fea = vis_fea.unsqueeze(1) + word_embs
            word_fused_gloss_logits = self.word_fused_gloss_output_layer(fused_fea)

        return {'gloss_feature': None,
                'gloss_feature_norm': None,
                'gloss_logits': logits, 
                'split_logits': split_logits,
                'bag_logits': bag_logits,
                'gloss_probabilities': gloss_probabilities,
                'word_fused_gloss_logits': word_fused_gloss_logits,
                'word_fused_gloss_probabilities': word_fused_gloss_probabilities,
                'topk_idx': topk_idx,
                'cam': cam
                }


class MarginVisualHead(nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf)"""
    def __init__(self, cls_num, input_size=1024, scale=64.0, margin=0.5, **kwargs):
        super().__init__()
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.margin = margin
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (cls_num, input_size)))
        self.mean_first = kwargs.pop('mean_first', False)
        self.variant = kwargs.pop('variant', 'arcface')


    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        if x.ndim > 3:
            x = F.avg_pool3d(x, (2, x.size(3), x.size(4)), stride=1)  #spatial global average pool
            x = x.view(x.size(0), x.size(1), x.size(2)).permute(0, 2, 1)  #B,T,C
        if self.mean_first:
            x = x.mean(dim=1)
        logits = F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))
        if not self.mean_first:
            logits = logits.mean(dim=1)  #[B,C]
        raw_logits = logits * self.scale
        
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        if self.variant == 'arcface':
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            final_target_logit = torch.where(
                    target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)
        elif self.variant == 'cosface':
            final_target_logit = torch.where(
                    target_logit > self.margin, target_logit - self.margin, target_logit)
            # final_target_logit = target_logit - self.margin

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale

        return {'gloss_feature': x,
                'gloss_feature_norm': F.normalize(x, dim=-1),
                'fea_vect': None,
                'gloss_logits': logits, 
                'gloss_raw_logits': raw_logits,
                'gloss_probabilities': raw_logits.softmax(-1),
                'word_emb_att_scores': None}


class WeightLearner(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, num_inputs=3):
        super().__init__()
        self.num_inputs = num_inputs
        self.input_size = input_size
        self.att_layer_lst = nn.ModuleList()
        for i in range(num_inputs):
            inp_size = input_size if i<num_inputs-1 else input_size*2
            self.att_layer_lst.append(nn.Sequential(
                nn.Linear(inp_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(hidden_size, 1)
            ))
    
    def forward(self, fea_lst):
        assert len(fea_lst) == self.num_inputs
        weights = []
        for i in range(len(fea_lst)):
            weights.append(self.att_layer_lst[i](fea_lst[i].mean(dim=1)))
        weights = torch.cat(weights, dim=-1)  #[B,3]
        weights = weights.softmax(dim=-1)
        return weights


class VisualHead(torch.nn.Module):
    def __init__(self, cls_num, input_size=832, hidden_size=512, ff_size=2048, pe=True,
                ff_kernelsize=3, pretrained_ckpt=None, is_empty=False, frozen=False, **kwargs):
        super().__init__()
        self.is_empty = is_empty
        if not is_empty:
            self.frozen = frozen
            self.hidden_size = hidden_size

            if input_size is None:
                self.fc1 = nn.Identity()
            else:
                self.fc1 = torch.nn.Linear(input_size, self.hidden_size)
            # self.bn1 = MaskedNorm(num_features=self.hidden_size, norm_type='sync_batch')
            self.bn1 = torch.nn.SyncBatchNorm(num_features=hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(p=0.1)

            if pe:
                self.pe = PositionalEncoding(self.hidden_size)
            else:
                self.pe = torch.nn.Identity()

            self.feedforward = PositionwiseFeedForward(input_size=self.hidden_size, ff_size=ff_size,
                                                        dropout=0.1, kernel_size=ff_kernelsize, skip_connection=True)
            self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)
            self.gloss_output_layer = torch.nn.Linear(self.hidden_size, cls_num)

            if self.frozen:
                self.frozen_layers = [self.fc1, self.bn1, self.relu1, self.pe, self.dropout1, self.feedforward, self.layer_norm]
                for layer in self.frozen_layers:
                    for name, param in layer.named_parameters():
                        param.requires_grad = False
                    layer.eval()
        else:
            self.gloss_output_layer = torch.nn.Linear(input_size, cls_num)
        if pretrained_ckpt:
            self.load_from_pretrained_ckpt(pretrained_ckpt)
    
    
    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        logger = get_logger()
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {}
        for k,v in checkpoint.items():
            if 'recognition_network.visual_head.' in k:
                load_dict[k.replace('recognition_network.visual_head.','')] = v
        self.load_state_dict(load_dict)
        logger.info('Load Visual Head from pretrained ckpt {}'.format(pretrained_ckpt))


    def forward(self, x):
        if x.ndim > 3:
            x = x.mean(dim=(-2,-1)).permute(0,2,1)
        B, Tin, D = x.shape 
        if not self.is_empty:
            if not self.frozen:
                #projection 1
                x = self.fc1(x)
                x = x.view(-1, self.hidden_size)
                x = self.bn1(x)
                x = x.view(B, Tin, self.hidden_size)
                x = self.relu1(x)
                #pe
                x = self.pe(x)
                x = self.dropout1(x)

                #feedforward
                x = self.feedforward(x)
                x = self.layer_norm(x)

            else:
                with torch.no_grad():
                    for ii, layer in enumerate(self.frozen_layers):
                        layer.eval()
                        if ii==1:
                            x = layer(x)
                        else:
                            x = layer(x)

        logits = self.gloss_output_layer(x) #B,T,V
        logits = logits.mean(dim=1)  #B,V, same as S3D
        gloss_probabilities = logits.softmax(1)

        return {'gloss_feature': x,
                'gloss_feature_norm': F.normalize(x, dim=-1),
                'gloss_logits': logits, 
                'gloss_probabilities': gloss_probabilities}