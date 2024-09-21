import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], channel_spatial=[False, True]):
        super(CBAM, self).__init__()
        self.channel_spatial = channel_spatial
        if channel_spatial[0]:
            self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        if channel_spatial[1]:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        if self.channel_spatial[0]:
            x_out = self.ChannelGate(x)
        if self.channel_spatial[1]:
            x_out = self.SpatialGate(x)
        return x_out


#----------------------------------------------Two stream adaptation-------------------------------------------
class ChannelPool_twostream(nn.Module):
    def forward(self, x_rgb, x_pose):
        return torch.cat( (torch.max(x_rgb,1)[0].unsqueeze(1), torch.mean(x_rgb,1).unsqueeze(1), torch.max(x_pose,1)[0].unsqueeze(1), torch.mean(x_pose,1).unsqueeze(1)), dim=1 )

class SpatialGate_twostream(nn.Module):
    def __init__(self):
        super(SpatialGate_twostream, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool_twostream()
        self.spatial = BasicConv(4, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x_rgb, x_pose):
        # x_rgb, x_pose [NT,C,H,W]
        x_compress = self.compress(x_rgb, x_pose)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting, [NT,1,H,W]
        return scale, x_rgb*scale + x_pose*(1.0-scale)

class CBAM_twostream(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], channel_spatial=[False, True]):
        super(CBAM_twostream, self).__init__()
        self.channel_spatial = channel_spatial
        if channel_spatial[0]:
            self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        if channel_spatial[1]:
            self.SpatialGate = SpatialGate_twostream()
    def forward(self, x_rgb, x_pose):
        N,C,T,H,W = x_rgb.shape
        x_rgb = x_rgb.view(-1,C,H,W)
        x_pose = x_pose.view(-1,C,H,W)
        if self.channel_spatial[0]:
            x_out = self.ChannelGate(x_rgb, x_pose)
        if self.channel_spatial[1]:
            spat_gate, x_out = self.SpatialGate(x_rgb, x_pose)
        return spat_gate.view(N,1,T,H,W), x_out.view(N,C,T,H,W)