import torch
import torch.nn as nn
from modelling.transformer.encoders import TransformerEncoder
from modelling.transformer.layers import PositionalEncoding


class TrajModel(nn.Module):
    def __init__(self, init_planes, dim=512, num_layers=3, seq_model_type='transformer', **kwargs):
        super().__init__()
        self.dim = dim
        self.mapper = nn.Linear(init_planes, dim)
        self.seq_model_type = seq_model_type
        if seq_model_type == 'transformer':
            self.seq_model = TransformerEncoder(hidden_size=dim, ff_size=2048, num_layers=num_layers, num_heads=8, output_size=dim)

        elif 'transformer_pos' in seq_model_type:
            self.mapper = nn.Identity()
            self.seq_model_intra = TransformerEncoder(hidden_size=dim, ff_size=2048, num_layers=num_layers, num_heads=8, output_size=dim)
            self.seq_model_inter = TransformerEncoder(hidden_size=dim, ff_size=2048, num_layers=num_layers, num_heads=8, output_size=dim)
            hmap_cfg = kwargs.pop('heatmap_cfg', {})
            self.max_h = self.max_w = hmap_cfg['input_size']
            if seq_model_type == 'transformer_pos':
                self.x_emb = nn.Embedding(num_embeddings=self.max_w, embedding_dim=dim//2)
                self.y_emb = nn.Embedding(num_embeddings=self.max_h, embedding_dim=dim//2)
            elif seq_model_type == 'transformer_pos_cos':
                self.cos_pe = PositionalEncoding(size=dim//2, max_len=self.max_h)

        elif seq_model_type == 'tcn':
            mod_lst = []
            for _ in range(num_layers):
                mod_lst.append(nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))
                mod_lst.append(nn.BatchNorm1d(dim))
                mod_lst.append(nn.ReLU(inplace=True))
            self.seq_model = nn.Sequential(*mod_lst)
        else:
            raise ValueError
    
    def forward(self, x):
        x = self.mapper(x)
        if self.seq_model_type == 'transformer':
            x, _, _, _ = self.seq_model(x, mask=None, output_attention=False)
        elif 'transformer_pos' in self.seq_model_type:
            x = 0.5 + x*0.5
            x[..., 1] = x[..., 1] * self.max_h
            x[..., 0] = x[..., 0] * self.max_w
            B,T = x.shape[:2]
            x = x.reshape(B,T,-1,2)
            c_x, c_y = x.split(1, dim=-1)
            c_x, c_y = torch.clamp(c_x, 0, self.max_w-1).squeeze(-1).long(), torch.clamp(c_y, 0, self.max_h-1).squeeze(-1).long()

            if self.seq_model_type == 'transformer_pos':
                x_fea = self.x_emb(c_x)
                y_fea = self.y_emb(c_y)
            elif self.seq_model_type == 'transformer_pos_cos':
                c_x, c_y = c_x.reshape(-1), c_y.reshape(-1)
                x_fea, y_fea = self.cos_pe.pe.squeeze(0).index_select(dim=0, index=c_x), self.cos_pe.pe.squeeze(0).index_select(dim=0, index=c_y)
                x_fea, y_fea = x_fea.reshape(B,T,-1,self.dim//2), y_fea.reshape(B,T,-1,self.dim//2)

            x = torch.cat([x_fea, y_fea], dim=-1)
            B,T,K,C = x.shape
            x = x.reshape(-1, K, C)
            x, _, _, _ = self.seq_model_intra(x, mask=None, output_attention=False)
            x = x.mean(dim=1).reshape(B,T,C)
            x, _, _, _ = self.seq_model_inter(x, mask=None, output_attention=False)
        elif self.seq_model_type == 'tcn':
            x = self.seq_model(x.transpose(1,2))
            x = x.transpose(1,2)
        return {'sgn_feature': x}