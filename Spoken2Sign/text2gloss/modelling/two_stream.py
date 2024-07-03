import torch
from torch import nn
import torch.nn.functional as F

from modelling.fusion import Lateral_Conn
from modelling.S3D import S3D_backbone
from modelling.pyramid import PyramidNetwork


class S3D_two_stream_v2(nn.Module):
    # use pyramid v2
    def __init__(self, use_block=4, freeze_block=(1,0), downsample=2, pose_inchannels=17, flag_lateral=[False, False], **kwargs):
        super(S3D_two_stream_v2, self).__init__()
        # need to set channel dimension
        # NOTE: need to take lateral connections into consideration
        self.cfg_pyramid = kwargs.pop('cfg_pyramid', None)
        rgb_input_channels = kwargs.pop('rgb_input_channels', 3)
        self.rgb_stream = S3D_backbone(rgb_input_channels, use_block, freeze_block[0], downsample, cfg_pyramid=self.cfg_pyramid)
        self.pose_stream = S3D_backbone(pose_inchannels, use_block, freeze_block[1], downsample, cfg_pyramid=self.cfg_pyramid)
        self.use_block = use_block
        self.flag_lateral = flag_lateral
        lateral_variant = kwargs.pop('lateral_variant', [None, None])
        lateral_kszie = kwargs.pop('lateral_ksize', (7,3,3))
        lateral_ratio = kwargs.pop('lateral_ratio', (1,2,2))
        # lateral_interpolate = kwargs.pop('lateral_interpolate', False)
        self.fusion_features = kwargs.pop('fusion_features', ['c1','c2','c3'])
        if self.cfg_pyramid is not None:
            if self.cfg_pyramid['version'] == 'v2':
                self.init_levels = self.cfg_pyramid.get('num_levels', 3)
            else:
                self.init_levels = self.cfg_pyramid.get('num_levels', 4)

        # identify the index of each stage(block)
        # NOTE: As posec3d, no fusion in the final stage
        self.block_idx = [0, 3, 6, 12]  #block outputs index
        assert use_block == 4
        if len(self.fusion_features) == 2:
            self.fuse_idx = [3,6]
            inoutchannels = [(192,192), (480,480)]
        elif len(self.fusion_features) == 3 and self.fusion_features != ['p2','p3','c4']:
            self.fuse_idx = [0,3,6]
            inoutchannels = [(64,64), (192,192), (480,480)]
        elif len(self.fusion_features) == 3 and self.fusion_features == ['p2','p3','c4']:
            self.fuse_idx = [3,6,12]
            inoutchannels = [(192,192), (480,480), (832,832)]
        elif len(self.fusion_features) == 4:
            self.fuse_idx = [0,3,6,12]
            inoutchannels = [(64,64), (192,192), (480,480), (832,832)]
        else:
            raise ValueError
        
        if flag_lateral[0]:
            # pose2rgb
            self.rgb_stream_lateral = nn.ModuleList([Lateral_Conn(inoutchannels[i][0], inoutchannels[i][1], lateral_kszie, lateral_ratio, 'pose2rgb', lateral_variant[0])\
                                                     for i in range(len(inoutchannels))])
        if flag_lateral[1]:
            # rgb2pose
            self.pose_stream_lateral = nn.ModuleList([Lateral_Conn(inoutchannels[i][0], inoutchannels[i][1], lateral_kszie, lateral_ratio, 'rgb2pose', lateral_variant[1])\
                                                     for i in range(len(inoutchannels))])

        
    def forward(self, x_rgb, x_pose, sgn_lengths=None):
        B, C, T_in, H, W = x_rgb.shape
        rgb_fea_lst, pose_fea_lst = [], []
        for i, (rgb_layer, pose_layer) in enumerate(zip(self.rgb_stream.backbone.base, self.pose_stream.backbone.base)):
            x_rgb = rgb_layer(x_rgb)
            x_pose = pose_layer(x_pose)
            if i in self.fuse_idx:
                x_rgb_fused = x_rgb
                x_pose_fused = x_pose

                if self.flag_lateral[0] and 'p' not in self.fusion_features[self.fuse_idx.index(i)]:
                    # print(self.fusion_features[self.fuse_idx.index(i)])
                    _, x_rgb_fused = self.rgb_stream_lateral[self.fuse_idx.index(i)](x_rgb, x_pose)
                if self.flag_lateral[1] and 'p' not in self.fusion_features[self.fuse_idx.index(i)]:
                    _, x_pose_fused = self.pose_stream_lateral[self.fuse_idx.index(i)](x_rgb, x_pose)
                
                x_rgb = x_rgb_fused
                x_pose = x_pose_fused

            if i in self.block_idx:
                rgb_fea_lst.append(x_rgb)
                pose_fea_lst.append(x_pose)

        rgb_fused = pose_fused = None
        diff = 1
        if self.cfg_pyramid['rgb'] or self.cfg_pyramid['pose']:
            num_levels = len(rgb_fea_lst)
            diff = num_levels - self.init_levels
            for i in range(num_levels-1, diff, -1):
                if self.rgb_stream.pyramid:
                    if self.cfg_pyramid['version'] == 'v2':
                        rgb_fea_lst[i-1] = rgb_fea_lst[i-1] + self.rgb_stream.pyramid.upsample_layers[num_levels-i-1](rgb_fea_lst[i])
                        rgb_fea_lst[i-1] = self.rgb_stream.pyramid.conv1x1_layers[num_levels-i-1](rgb_fea_lst[i-1])
                    else:
                        rgb_fea_lst[i-1] = rgb_fea_lst[i-1] + self.rgb_stream.pyramid.upsample_layers[num_levels-i-1](rgb_fea_lst[i])

                if self.pose_stream.pyramid:
                    if self.cfg_pyramid['version'] == 'v2':
                        pose_fea_lst[i-1] = pose_fea_lst[i-1] + self.pose_stream.pyramid.upsample_layers[num_levels-i-1](pose_fea_lst[i])
                        pose_fea_lst[i-1] = self.pose_stream.pyramid.conv1x1_layers[num_levels-i-1](pose_fea_lst[i-1])
                    else:
                        pose_fea_lst[i-1] = pose_fea_lst[i-1] + self.pose_stream.pyramid.upsample_layers[num_levels-i-1](pose_fea_lst[i])

                if 'p'+str(i) in self.fusion_features:
                    # print('p'+str(i))
                    _, rgb_fea_lst[i-1] = self.rgb_stream_lateral[self.fusion_features.index('p'+str(i))](rgb_fea_lst[i-1], pose_fea_lst[i-1])
                    _, pose_fea_lst[i-1] = self.pose_stream_lateral[self.fusion_features.index('p'+str(i))](rgb_fea_lst[i-1], pose_fea_lst[i-1])

        for i in range(len(rgb_fea_lst)):
            rgb_fea_lst[i] = rgb_fea_lst[i].mean(dim=(-2,-1)).permute(0,2,1)#B,C,T,H,W -> B,C,T -> B,T,C
            pose_fea_lst[i] = pose_fea_lst[i].mean(dim=(-2,-1)).permute(0,2,1)

        sgn_mask_lst, valid_len_out_lst = [], []
        rgb_out = pose_out = None
        if self.cfg_pyramid['rgb'] is None:
            rgb_out = rgb_fea_lst[-1]
        if self.cfg_pyramid['pose'] is None:
            pose_out = pose_fea_lst[-1]
        for fea in pose_fea_lst:
            #might be unpooled
            B, T_out, _ = fea.shape
            sgn_mask = torch.zeros([1,1,T_out], dtype=torch.bool, device=fea.device)  #modify B to 1
            valid_len_out = torch.floor(sgn_lengths*T_out/T_in).long() #B,
            # print(fea.shape, valid_len_out)
            for bi in range(1):
                sgn_mask[bi, :, :valid_len_out[bi]] = True
            sgn_mask_lst.append(sgn_mask)
            valid_len_out_lst.append(valid_len_out)

        return {'sgn_feature': rgb_out, 'pose_feature': pose_out, 'sgn_mask': sgn_mask_lst[diff:], 'valid_len_out': valid_len_out_lst[diff:],
                'rgb_fused': rgb_fused, 'pose_fused': pose_fused, 'rgb_fea_lst': rgb_fea_lst[diff:], 'pose_fea_lst': pose_fea_lst[diff:]}



class S3D_two_stream(nn.Module):
    def __init__(self, use_block=4, freeze_block=(1,0), downsample=2, pose_inchannels=17, flag_lateral=[False, False], **kwargs):
        super(S3D_two_stream, self).__init__()
        # need to set channel dimension
        # NOTE: need to take lateral connections into consideration
        self.rgb_stream = S3D_backbone(3, use_block, freeze_block[0], downsample)
        self.pose_stream = S3D_backbone(pose_inchannels, use_block, freeze_block[1], downsample)
        self.use_block = use_block
        self.flag_lateral = flag_lateral
        lateral_variant = kwargs.pop('lateral_variant', [None, None])
        lateral_kszie = kwargs.pop('lateral_ksize', (7,3,3))
        self.flag_pyramid = kwargs.pop('flag_pyramid', [None, None])

        # identify the index of each stage(block)
        # NOTE: As posec3d, no fusion in the final stage
        self.stage_idx = [0, 3, 6, 12]
        self.stage_idx = self.stage_idx[:use_block]

        # need to know inchannels and outchannels for each lateral connection
        inoutchannels_rgb = [(64,64), (192,192), (480,480), (832,832)]
        inoutchannels_pose = [(64,64), (192,192), (480,480), (832,832)]
        inoutchannels_rgb = inoutchannels_rgb[:use_block-1]
        inoutchannels_pose = inoutchannels_pose[:use_block-1]
        
        if flag_lateral[0]:
            # pose2rgb
            self.rgb_stream_lateral = nn.ModuleList([Lateral_Conn(inoutchannels_rgb[i][0], inoutchannels_rgb[i][1], lateral_kszie, (1,2,2), 'pose2rgb', lateral_variant[0])\
                                                     for i in range(use_block-1)])
        if flag_lateral[1]:
            # rgb2pose
            self.pose_stream_lateral = nn.ModuleList([Lateral_Conn(inoutchannels_pose[i][0], inoutchannels_pose[i][1], lateral_kszie, (1,2,2), 'rgb2pose', lateral_variant[1])\
                                                     for i in range(use_block-1)])
        
        if self.flag_pyramid[0]:
            self.rgb_pyramid = PyramidNetwork(channels=[832,480,192,64], kernel_size=3, num_levels=use_block, temp_scale=[2,1,1], spat_scale=[2,2,2])
        if self.flag_pyramid[1]:
            self.pose_pyramid = PyramidNetwork(channels=[832,480,192,64], kernel_size=3, num_levels=use_block, temp_scale=[2,1,1], spat_scale=[2,2,2])
        
        
    def forward(self, x_rgb, x_pose, sgn_lengths=None):
        B, C, T_in, H, W = x_rgb.shape
        rgb_fea_lst, pose_fea_lst = [], []
        for i, (rgb_layer, pose_layer) in enumerate(zip(self.rgb_stream.backbone.base, self.pose_stream.backbone.base)):
            x_rgb = rgb_layer(x_rgb)
            x_pose = pose_layer(x_pose)
            if i in self.stage_idx[:self.use_block-1]:
                x_rgb_fused = x_rgb
                x_pose_fused = x_pose

                if self.flag_lateral[0]:
                    _, x_rgb_fused = self.rgb_stream_lateral[self.stage_idx.index(i)](x_rgb, x_pose)
                if self.flag_lateral[1]:
                    _, x_pose_fused = self.pose_stream_lateral[self.stage_idx.index(i)](x_rgb, x_pose)
                
                x_rgb = x_rgb_fused
                x_pose = x_pose_fused

                rgb_fea_lst.append(x_rgb)
                pose_fea_lst.append(x_pose)

            if i == self.stage_idx[-1]:
                rgb_fea_lst.append(x_rgb)
                pose_fea_lst.append(x_pose)

        rgb_fused = pose_fused = None
        if self.flag_pyramid[0]:
            rgb_fea_lst, rgb_fused = self.rgb_pyramid(rgb_fea_lst, need_fused=True if 'fused' in self.flag_pyramid[0] else False)
            assert len(rgb_fea_lst) == self.use_block
        if self.flag_pyramid[1]:
            pose_fea_lst, pose_fused = self.pose_pyramid(pose_fea_lst, need_fused=True if 'fused' in self.flag_pyramid[1] else False)
            assert len(pose_fea_lst) == self.use_block

        sgn_mask_lst, valid_len_out_lst = [], []
        rgb_out = pose_out = None
        if self.flag_pyramid[0] is None and self.flag_pyramid[1] is None:
            B, _, T_out, _, _ = x_rgb.shape

            pooled_rgb_feature = torch.mean(x_rgb, dim=[3,4]) #B, D, T_out
            rgb_out = torch.transpose(pooled_rgb_feature, 1, 2) #b, t_OUT, d

            pooled_pose_feature = torch.mean(x_pose, dim=[3,4]) #B, D, T_out
            pose_out = torch.transpose(pooled_pose_feature, 1, 2) #b, t_OUT, d
        
            sgn_mask = torch.zeros([B,1,T_out], dtype=torch.bool, device=x_rgb.device)
            valid_len_out = torch.floor(sgn_lengths*T_out/T_in).long() #B,
            for bi in range(B):
                sgn_mask[bi, :, :valid_len_out[bi]] = True
            sgn_mask_lst.append(sgn_mask)
            valid_len_out_lst.append(valid_len_out)
        
        elif self.flag_pyramid[0] is None and self.flag_pyramid[1] is not None:
            pooled_rgb_feature = torch.mean(x_rgb, dim=[3,4]) #B, D, T_out
            rgb_out = torch.transpose(pooled_rgb_feature, 1, 2) #b, t_OUT, d

            for fea in pose_fea_lst:
                B, T_out, _ = fea.shape
                sgn_mask = torch.zeros([B,1,T_out], dtype=torch.bool, device=fea.device)
                valid_len_out = torch.floor(sgn_lengths*T_out/T_in).long() #B,
                for bi in range(B):
                    sgn_mask[bi, :, :valid_len_out[bi]] = True
                sgn_mask_lst.append(sgn_mask)
                valid_len_out_lst.append(valid_len_out)
        
        elif self.flag_pyramid[0] is not None and self.flag_pyramid[1] is None:
            pooled_pose_feature = torch.mean(x_pose, dim=[3,4]) #B, D, T_out
            pose_out = torch.transpose(pooled_pose_feature, 1, 2) #b, t_OUT, d

            for fea in rgb_fea_lst:
                B, T_out, _ = fea.shape
                sgn_mask = torch.zeros([B,1,T_out], dtype=torch.bool, device=fea.device)
                valid_len_out = torch.floor(sgn_lengths*T_out/T_in).long() #B,
                for bi in range(B):
                    sgn_mask[bi, :, :valid_len_out[bi]] = True
                sgn_mask_lst.append(sgn_mask)
                valid_len_out_lst.append(valid_len_out)

        else:
            for fea in pose_fea_lst:
                B, T_out, _ = fea.shape
                sgn_mask = torch.zeros([B,1,T_out], dtype=torch.bool, device=fea.device)
                valid_len_out = torch.floor(sgn_lengths*T_out/T_in).long() #B,
                for bi in range(B):
                    sgn_mask[bi, :, :valid_len_out[bi]] = True
                sgn_mask_lst.append(sgn_mask)
                valid_len_out_lst.append(valid_len_out)

        return {'sgn_feature': rgb_out, 'pose_feature': pose_out, 'sgn_mask': sgn_mask_lst, 'valid_len_out': valid_len_out_lst,
                'rgb_fused': rgb_fused, 'pose_fused': pose_fused, 'rgb_fea_lst': rgb_fea_lst, 'pose_fea_lst': pose_fea_lst}