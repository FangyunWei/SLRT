import torch as t
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, **kwargs):
        super(FCN, self).__init__()
        input_channels = 3
        self.CNN_stack = nn.ModuleList([
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
            ])

        self.TCN_stack = nn.ModuleList([
            nn.Conv1d(512, 512, 5, 1, 2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(512, 512, 5, 1, 2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(512, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        ])
    
    def set_train(self):
        self.train()

    def get_frozen_layers(self):
        return []
    
    def set_frozen_layers(self):
        for m in getattr(self, 'frozen_modules', []):
            for param in m.parameters():
                #print(param)
                param.requires_grad = False
            m.eval()

    def forward(self, sgn_videos, sgn_lengths=None):
        B,C,T_in,H,W = sgn_videos.shape
        x = sgn_videos.permute(0,2,1,3,4).view(-1,C,H,W)  #B*T,C,H,W

        for layer in self.CNN_stack:
            x = layer(x)
        x = t.flatten(x, 1)  #B*T,C
        x = x.view(B,-1,512).permute(0,2,1)  #B,C,T
        
        for layer in self.TCN_stack:
            x = layer(x)
        
        sgn_mask_lst, valid_len_out_lst = [], []
        B, C, T_out = x.shape
        sgn_mask = t.zeros([B,1,T_out], dtype=t.bool, device=x.device)
        valid_len_out = t.floor(sgn_lengths*T_out/T_in).long() #B,
        for bi in range(B):
            sgn_mask[bi, :, :valid_len_out[bi]] = True
        sgn_mask_lst.append(sgn_mask)
        valid_len_out_lst.append(valid_len_out)
        
        return {'sgn_feature':x.permute(0,2,1), 'sgn_mask':sgn_mask_lst, 
                'valid_len_out': valid_len_out_lst, 'fea_lst': [], 'sgn':x.permute(0,2,1)}
