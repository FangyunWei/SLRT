import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ConvModule(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            kernel_size,
            stride,
            padding,
            bias=False,
            groups=1,
    ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class LevelFusion(nn.Module):
    def __init__(self,
                 in_channels=[1024, 1024],
                 mid_channels=[1024, 1024],
                 out_channels=2048,
                 ds_scales=[(1, 1, 1), (1, 1, 1)],
                 ):
        super(LevelFusion, self).__init__()
        self.ops = nn.ModuleList()
        num_ins = len(in_channels)
        for i in range(num_ins):
            op = Temporal_Downsampling(in_channels[i], mid_channels[i], kernel_size=(1, 1, 1), stride=(1, 1, 1),
                             padding=(0, 0, 0), bias=False, groups=32, norm=True, activation=True,
                             downsample_position='before', downsample_scale=ds_scales[i])
            self.ops.append(op)

        in_dims = np.sum(mid_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_dims, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        out = [self.ops[i](feature) for i, feature in enumerate(inputs)]
        out = torch.cat(out, 1)
        out = self.fusion_conv(out)
        return out


class SpatialModulation(nn.Module):
    def __init__(
            self,
            inplanes=[1024, 2048],
            planes=2048,
    ):
        super(SpatialModulation, self).__init__()

        self.spatial_modulation = nn.ModuleList()
        for i, dim in enumerate(inplanes):
            op = nn.ModuleList()
            ds_factor = planes // dim
            ds_num = int(np.log2(ds_factor))
            if ds_num < 1:
                op = nn.Identity()
            else:
                for dsi in range(ds_num):
                    in_factor = 2 ** dsi
                    out_factor = 2 ** (dsi + 1)
                    op.append(ConvModule(dim * in_factor, dim * out_factor, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                         padding=(0, 1, 1), bias=False))
            self.spatial_modulation.append(op)

    def forward(self, inputs):
        out = []
        for i, feature in enumerate(inputs):
            if isinstance(self.spatial_modulation[i], nn.ModuleList):
                out_ = inputs[i]
                for III, op in enumerate(self.spatial_modulation[i]):
                    out_ = op(out_)
                out.append(out_)
            else:
                out.append(self.spatial_modulation[i](inputs[i]))
        return out


class Upsampling(nn.Module):
    def __init__(self, in_channels=832, out_channels=480, kernel_size=3, scale=(2,2,2), interpolate=True, adapt_first=False):
        super(Upsampling, self).__init__()
        self.scale = scale
        padding_s = (0, kernel_size//2, kernel_size//2)
        padding_t = (kernel_size//2, 0, 0)
        out_padding_s = (0,0,0)
        out_padding_t = (0,0,0)
        if scale[1] == 2:
            out_padding_s = (0,1,1)
        if scale[0] == 2:
            out_padding_t = (1,0,0)
        
        if adapt_first:
            # since pose branch input size is 112, for the last block (first deconv layer) we need to upsample 3 to 7.
            self.scale = scale = (2,3,3)
            out_padding_s = (0,0,0)
            out_padding_t = (1,0,0)
        
        self.interpolate = interpolate
        if interpolate:
            self.conv1x1 = ConvModule(in_channels, out_channels, 1, 1, 0, bias=False)
        else:
            self.conv_trans_s = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1,kernel_size,kernel_size),
                                            stride=(1,scale[1],scale[2]), padding=padding_s, output_padding=out_padding_s, bias=False)
            self.bn_s = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001, affine=True)
            self.relu_s = nn.ReLU(inplace=True)

            self.conv_trans_t = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(kernel_size,1,1),
                                            stride=(scale[0],1,1), padding=padding_t, output_padding=out_padding_t, bias=False)
            self.bn_t = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001, affine=True)
            self.relu_t = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.interpolate:
            x = F.interpolate(x, scale_factor=self.scale, mode='trilinear')
            x = self.conv1x1(x)
        else:
            x = self.conv_trans_s(x)
            x = self.bn_s(x)
            x = self.relu_s(x)

            x = self.conv_trans_t(x)
            x = self.bn_t(x)
            x = self.relu_t(x)
        return x


class Temporal_Downsampling(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, scale=2):
        super(Temporal_Downsampling, self).__init__()
        assert kernel_size % 2 == 1
        self.scale = scale
        padding_t = (kernel_size//2, 0, 0)

        self.conv_trans_t = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size,1,1),
                                    stride=(scale,1,1), padding=padding_t, bias=False)
        self.bn_t = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_trans_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x


class PyramidNetwork_v2(nn.Module):
    def __init__(self, channels=[832,480,192], kernel_size=1, num_levels=3, temp_scale=[2,1], spat_scale=[2,2]):
        super(PyramidNetwork_v2, self).__init__()
        self.num_levels = num_levels

        self.upsample_layers = nn.ModuleList()
        self.conv1x1_layers = nn.ModuleList()
        for i in range(num_levels-1):
            self.upsample_layers.append(Upsampling(channels[i], channels[i+1], kernel_size, scale=(temp_scale[i], spat_scale[i], spat_scale[i]), interpolate=True))
            self.conv1x1_layers.append(ConvModule(channels[i+1], channels[i+1], 1, 1, 0, bias=False))

    
    def forward(self, fea_lst, need_fused=False):
        # fea_lst: 1st layer output, 2nd layer output, ...
        assert len(fea_lst) == self.num_levels

        # top-down flow
        for i in range(self.num_levels-1, 0, -1):
            fea_lst[i-1] = fea_lst[i-1] + self.upsample_layers[self.num_levels-i-1](fea_lst[i])
            fea_lst[i-1] = self.conv1x1_layers[self.num_levels-i-1](fea_lst[i-1])

        for i in range(self.num_levels):
            fea_lst[i] = fea_lst[i].mean(dim=(-2,-1)).permute(0,2,1)

        return fea_lst, None


class PyramidNetwork(nn.Module):
    def __init__(self, channels=[1024,832,480,192,64], kernel_size=3, num_levels=4, temp_scale=[2,2,1,1], spat_scale=[2,2,2,2], need_fused=False, adapt_first=False):
        super(PyramidNetwork, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_levels = num_levels
        self.temp_scale = temp_scale
        self.spat_scale = spat_scale

        self.upsample_layers = nn.ModuleList()
        if need_fused:
            self.temp_downsample_layers = nn.ModuleList()
        for i in range(num_levels-1):
            self.upsample_layers.append(Upsampling(channels[i], channels[i+1], kernel_size, scale=(temp_scale[i], spat_scale[i], spat_scale[i]), interpolate=False,
                                        adapt_first=adapt_first if i==0 else False))
            if need_fused:
                self.temp_downsample_layers.append(Temporal_Downsampling(channels[num_levels-1-i], channels[num_levels-1-i], kernel_size, scale=2))

    
    def forward(self, fea_lst, need_fused=False):
        # fea_lst: 1st layer output, 2nd layer output, ...
        assert len(fea_lst) == self.num_levels

        # top-down flow
        for i in range(self.num_levels-1, 0, -1):
            fea_lst[i-1] = fea_lst[i-1] + self.upsample_layers[self.num_levels-i-1](fea_lst[i])

        fused = []
        for i in range(self.num_levels):
            if need_fused:
                if i != self.num_levels - 1:
                    fused.append(self.temp_downsample_layers[i](fea_lst[i]).mean(dim=(-2,-1)).permute(0,2,1))
                else:
                    # feature of the last layer don't need downsample
                    fused.append(fea_lst[i].mean(dim=(-2,-1)).permute(0,2,1))
            fea_lst[i] = fea_lst[i].mean(dim=(-2,-1)).permute(0,2,1)

        # get fused prediction
        if need_fused:
            fused = torch.cat(fused, dim=2)  #[B,T,C]
        return fea_lst, fused