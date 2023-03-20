import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import get_logger
from modelling.utils import PositionalEncoding, MaskedNorm, PositionwiseFeedForward, MLPHead
class VisualHead(torch.nn.Module):
    def __init__(self, 
        cls_num, input_size=832, hidden_size=512, ff_size=2048, pe=True,
        ff_kernelsize=3, pretrained_ckpt=None, is_empty=False, frozen=False, 
        plus_conv_cfg={},
        ssl_projection_cfg={}):
        super().__init__()
        self.is_empty = is_empty
        self.plus_conv_cfg = plus_conv_cfg
        self.ssl_projection_cfg = ssl_projection_cfg
        if is_empty==False:
            self.frozen = frozen
            self.hidden_size = hidden_size

            if input_size is None:
                self.fc1 = nn.Identity()
            else:
                self.fc1 = torch.nn.Linear(input_size, self.hidden_size)
            self.bn1 = MaskedNorm(num_features=self.hidden_size, norm_type='sync_batch')
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(p=0.1)

            if pe:
                self.pe = PositionalEncoding(self.hidden_size)
            else:
                self.pe = torch.nn.Identity()

            self.feedforward = PositionwiseFeedForward(input_size=self.hidden_size,
                ff_size=ff_size,
                dropout=0.1, kernel_size=ff_kernelsize, skip_connection=True)
            
            self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)

            if plus_conv_cfg!={}:
                plus_convs = []
                for i in range(plus_conv_cfg['num_layer']):
                    plus_convs.append(nn.Conv1d(self.hidden_size, self.hidden_size, 
                        kernel_size=plus_conv_cfg['kernel_size'], stride=plus_conv_cfg['stride'], padding_mode='replicate'))
                self.plus_conv = nn.Sequential(*plus_convs)
            else:
                self.plus_conv = nn.Identity()

            if ssl_projection_cfg!={}:
                self.ssl_projection = MLPHead(embedding_size=self.hidden_size, 
                    projection_hidden_size=ssl_projection_cfg['hidden_size'])

            self.gloss_output_layer = torch.nn.Linear(self.hidden_size, cls_num)

            if self.frozen:
                self.frozen_layers = [self.fc1, self.bn1, self.relu1,  self.pe, self.dropout1, self.feedforward, self.layer_norm]
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

    def forward(self, x, mask, valid_len_in=None):
        B, Tin, D = x.shape 
        if self.is_empty==False:
            if not self.frozen:
                #projection 1
                x = self.fc1(x)
                x = self.bn1(x, mask)
                x = self.relu1(x)
                #pe
                x = self.pe(x)
                x = self.dropout1(x)

                #feedforward
                x = self.feedforward(x)
                x = self.layer_norm(x)

                x = x.transpose(1,2)
                x = self.plus_conv(x)
                x = x.transpose(1,2)
            else:
                with torch.no_grad():
                    for ii, layer in enumerate(self.frozen_layers):
                        layer.eval()
                        if ii==1:
                            x = layer(x, mask)
                        else:
                            x = layer(x)
                x = x.transpose(1,2)
                x = self.plus_conv(x)
                x = x.transpose(1,2)

        #classification
        logits = self.gloss_output_layer(x) #B,T,V
        gloss_probabilities_log = logits.log_softmax(2) 
        gloss_probabilities = logits.softmax(2)

        if self.plus_conv_cfg!={}:
            B, Tout, D = x.shape
            valid_len_out = torch.floor(valid_len_in*Tout/Tin).long() #B,
        else:
            valid_len_out = valid_len_in
        if self.ssl_projection_cfg!={}:
            x_ssl = self.ssl_projection(x)
            if self.ssl_projection_cfg['normalize']==True:
                x_ssl = F.normalize(x_ssl, dim=-1)
        else:
            x_ssl = None
        return {'gloss_feature_ssl':x_ssl, 
                'gloss_feature': x,
                'gloss_feature_norm': F.normalize(x, dim=-1),
                'gloss_logits':logits, 
                'gloss_probabilities_log':gloss_probabilities_log,
                'gloss_probabilities': gloss_probabilities,
                'valid_len_out':valid_len_out}