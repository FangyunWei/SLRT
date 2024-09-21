import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import LabelSmoothCE


class BagDenoiser(nn.Module):
    def __init__(self, cfg, cls_num=2000):
        super().__init__()
        self.cfg = cfg
        self.model_type = cfg.get('type', None)
        hidden_size = cfg.get('hidden_size', 512)
        self.context_size = cfg.get('context_size', 0)
        input_size = 1024*2
        if self.model_type == 'mlp':
            self.denoiser = nn.Sequential(nn.Linear(input_size, hidden_size),
                                          nn.ReLU(True),
                                          nn.Linear(hidden_size, input_size))

        elif self.model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=8, dim_feedforward=2048, batch_first=True)
            self.denoiser = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.cls_token = nn.Parameter(torch.randn(1, 1, input_size))

        self.fc = nn.Linear(input_size, cls_num)
        ignore_index = cfg.get('ignore_index', -100)
        self.loss_func = LabelSmoothCE(lb_smooth=0.2, ignore_index=ignore_index, reduction='mean')


    def decode(self, logits, k=1):
        res = torch.argsort(logits, dim=-1, descending=True)
        res = res[..., :k]
        return res


    def forward(self, features, labels=None, **kwargs):
        if self.model_type == 'mlp':
            x = features.mean(dim=1)  #B,C
            x = self.denoiser(x)
            logits = self.fc(x)
        
        elif self.model_type == 'transformer':
            B = features.shape[0]
            cls_token = self.cls_token.repeat(B,1,1)
            x = torch.cat([cls_token, features], dim=1)  #B,w+1,C
            x = self.denoiser(x)
            cls_emb = x[:,0,:]  #B,C
            if self.context_size > 0:
                cls_emb = cls_emb.transpose(0,1).unsqueeze(0)  #1,C,B
                cls_emb = F.avg_pool1d(cls_emb, kernel_size=self.context_size, stride=1, padding=self.context_size//2)
                cls_emb = cls_emb.squeeze(0).transpose(0,1)  #B,C
            logits = self.fc(cls_emb)

        loss = None
        if labels is not None:
            loss = self.loss_func(logits, labels)

        outputs = {'logits': logits,
                   'denoise_loss': loss}
        return outputs
        