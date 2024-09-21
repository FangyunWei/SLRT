from modelling.transformer.encoders import TransformerEncoder
import torch
import torch.nn as nn
from utils.misc import get_logger
from modelling.utils import PositionalEncoding, MaskedNorm, PositionwiseFeedForward

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(TemporalBlock, self).__init__()
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1d = torch.nn.BatchNorm1d(num_features=out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn1d(x)
        x = self.relu(x)
        return x

class IntermediateBlock(nn.Module):
    def __init__(self, in_channels, out_channels=512):
        super().__init__()
        self.fc = nn.Linear(in_channels, 512)
        self.norm = MaskedNorm(norm_type='sync_batch', num_features=512)
        self.activation = nn.ReLU()
    def forward(self, x, mask):
        x = self.fc(x)
        x = self.norm(x, mask)
        x = self.activation(x)
        return x

class VisualHeadNew(torch.nn.Module):
    def __init__(self, cls_num, input_size, temporal_block_layers, encoder, **kwargs):
        super().__init__()
        if temporal_block_layers==0:
            temporal_blocks = nn.Identity()
        else:
            temporal_blocks = [
                TemporalBlock(in_channels=input_size, out_channels=512, kernel_size=5, stride=1, padding=2),
                torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1)]
            for i in range(1, temporal_block_layers):
                temporal_blocks += [
                    TemporalBlock(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2),
                    torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1)]
            self.temporal_blocks = nn.Sequential(*temporal_blocks)

        #intermediate (SgnEmbed)
        self.intermediate = IntermediateBlock(in_channels=512 if temporal_block_layers>=1 else input_size)
       
        #encoder
        if encoder['type'] == 'transformer':
            self.encoder = TransformerEncoder(**encoder)
        else:
            raise ValueError
        
        self.gloss_output_layer = torch.nn.Linear(512, cls_num)
        return

    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        pass
    def forward(self, x, mask, valid_len_in):
        #1. temporal block (T->T/4)
        #print('x ', x.shape, 'mask ', mask.shape) 
        B, Tin, D = x.shape
        x = self.temporal_blocks(x.transpose(1,2))
        x = x.transpose(1,2)
        B, Tout, D = x.shape
        #valid_len_in = torch.sum(mask, dim=-1).unsqueeze(-1)#B,1,T -> B,1-> B
        downsampled_mask = torch.zeros([B,1,Tout], dtype=torch.bool, device=x.device)
        valid_len_out = torch.floor(valid_len_in*Tout/Tin).long() #B,
        for bi in range(B):
            downsampled_mask[:,:,:valid_len_out[bi]] = True
        #print(valid_len_in, valid_len_out)
            
        #print('after temporal block x', x.shape)

        #2. intermediate
        x = self.intermediate(x, downsampled_mask)

        #3. encoder
        x = self.encoder(embed_src=x, src_length=valid_len_out, mask=downsampled_mask)[0]

        #4. classification
        logits = self.gloss_output_layer(x) #B,T,V
        gloss_probabilities_log = logits.log_softmax(2) 
        gloss_probabilities = logits.softmax(2)
        return {'gloss_feature':x, 
                'gloss_logits':logits, 
                'gloss_probabilities_log':gloss_probabilities_log,
                'gloss_probabilities': gloss_probabilities,
                'valid_len_out': valid_len_out}