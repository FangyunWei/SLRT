from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import get_logger
import math
import numpy as np


class SepConvVisualHead(torch.nn.Module):
    def __init__(self, cls_num, input_size=1024, hidden_size=512, ff_size=2048, pe=True,
                ff_kernelsize=3, pretrained_ckpt=None, is_empty=False, frozen=False, **kwargs):
        super().__init__()
        self.frozen = frozen
        self.input_size = input_size
        self.topk = kwargs.pop('topk', 5)
        self.gloss_output_layer = nn.Linear(input_size, cls_num)
        
        self.contras_setting = kwargs.pop('contras_setting', 'dual_ema_cosine')
        if self.contras_setting is not None:
            self.word_emb_tab = kwargs.pop('word_emb_tab', None)
            self.word_emb_dim = kwargs.pop('word_emb_dim', 0)
            assert self.word_emb_tab is not None and self.word_emb_dim > 0
            self.word_emb_tab.requires_grad = False
            self.word_emb_sim = torch.matmul(F.normalize(self.word_emb_tab, dim=-1), F.normalize(self.word_emb_tab, dim=-1).T)
            self.word_emb_mapper = nn.Linear(self.word_emb_dim, input_size)
            if 'dual' in self.contras_setting:
                self.word_fused_gloss_output_layer = nn.Linear(input_size, cls_num)

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


    def forward(self, x, labels=None):
        if x.ndim > 3:
            x = F.avg_pool3d(x, (2, x.size(3), x.size(4)), stride=1)  #spatial global average pool
            x = x.view(x.size(0), x.size(1), x.size(2)).permute(0, 2, 1)  #B,T,C
        if x.ndim>2:
            x = x.mean(dim=1)

        B, C = x.shape[0], x.shape[-1]
        vis_fea = x

        # interact with word_embs
        k = gloss_probabilities = word_fused_gloss_logits = word_fused_gloss_probabilities = topk_idx = None

        logits = self.gloss_output_layer(x)  #[B,N]
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
                'gloss_probabilities': gloss_probabilities,
                'word_fused_gloss_logits': word_fused_gloss_logits,
                'word_fused_gloss_probabilities': word_fused_gloss_probabilities,
                'topk_idx': topk_idx
                }
