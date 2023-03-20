import torch
from torch import nn
import torch.nn.functional as F

from modelling.fusion import Lateral_Conn
from modelling.S3D import S3D_backbone


class S3D_two_stream_v2(nn.Module):
    def __init__(self, use_block=4, freeze_block=(1,0), downsample=2, pose_inchannels=17, flag_lateral=[False, False], **kwargs):
        super(S3D_two_stream_v2, self).__init__()
        # need to set channel dimension
        # NOTE: need to take lateral connections into consideration
        self.cfg_pyramid = kwargs.pop('cfg_pyramid', None)
        self.rgb_stream = S3D_backbone(3, use_block, freeze_block[0], downsample, cfg_pyramid=self.cfg_pyramid)
        self.pose_stream = S3D_backbone(pose_inchannels, use_block, freeze_block[1], downsample, cfg_pyramid=self.cfg_pyramid)
        self.use_block = use_block
        self.flag_lateral = flag_lateral
        lateral_kszie = kwargs.pop('lateral_ksize', (7,3,3))
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
            self.rgb_stream_lateral = nn.ModuleList([Lateral_Conn(inoutchannels[i][0], inoutchannels[i][1], lateral_kszie, (1,2,2), 'pose2rgb')\
                                                     for i in range(len(inoutchannels))])
        if flag_lateral[1]:
            # rgb2pose
            self.pose_stream_lateral = nn.ModuleList([Lateral_Conn(inoutchannels[i][0], inoutchannels[i][1], lateral_kszie, (1,2,2), 'rgb2pose')\
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
            rgb_fea_lst[i] = rgb_fea_lst[i].mean(dim=(-2,-1)).permute(0,2,1)
            pose_fea_lst[i] = pose_fea_lst[i].mean(dim=(-2,-1)).permute(0,2,1)

        sgn_mask_lst, valid_len_out_lst = [], []
        rgb_out = pose_out = None
        if self.cfg_pyramid['rgb'] is None:
            rgb_out = rgb_fea_lst[-1]
        if self.cfg_pyramid['pose'] is None:
            pose_out = pose_fea_lst[-1]
        for fea in pose_fea_lst:
            B, T_out, _ = fea.shape
            sgn_mask = torch.zeros([B,1,T_out], dtype=torch.bool, device=fea.device)
            valid_len_out = torch.floor(sgn_lengths*T_out/T_in).long() 
            for bi in range(B):
                sgn_mask[bi, :, :valid_len_out[bi]] = True
            sgn_mask_lst.append(sgn_mask)
            valid_len_out_lst.append(valid_len_out)

        return {'sgn_feature': rgb_out, 'pose_feature': pose_out, 'sgn_mask': sgn_mask_lst[diff:], 'valid_len_out': valid_len_out_lst[diff:],
                'rgb_fused': rgb_fused, 'pose_fused': pose_fused, 'rgb_fea_lst': rgb_fea_lst[diff:], 'pose_fea_lst': pose_fea_lst[diff:]}
