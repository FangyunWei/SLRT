import torch
from torch import nn
import torch.nn.functional as F
from modelling.fusion import Lateral_Conn
from modelling.S3D import S3D_backbone


class S3D_two_stream_v2(nn.Module):
    # use pyramid v2
    def __init__(self, use_block=5, freeze_block=(0,0), pose_inchannels=17, flag_lateral=[False, False], **kwargs):
        super(S3D_two_stream_v2, self).__init__()
        # need to set channel dimension
        # NOTE: need to take lateral connections into consideration
        self.cfg_pyramid = kwargs.pop('cfg_pyramid', None)
        self.use_shortcut = kwargs.pop('use_shortcut', False)
        self.rgb_stream = S3D_backbone(3, use_block, freeze_block[0], cfg_pyramid=self.cfg_pyramid, use_shortcut=self.use_shortcut)
        self.pose_stream = S3D_backbone(pose_inchannels, use_block, freeze_block[1], cfg_pyramid=self.cfg_pyramid, 
                                        coord_conv=kwargs.pop('coord_conv', None), use_shortcut=self.use_shortcut)
        self.use_block = use_block
        self.flag_lateral = flag_lateral
        lateral_variant = kwargs.pop('lateral_variant', [None, None])
        lateral_ksize = kwargs.pop('lateral_ksize', (1,3,3))
        lateral_ratio = kwargs.pop('lateral_ratio', (1,2,2))
        lateral_interpolate = kwargs.pop('lateral_interpolate', False)
        self.fusion_features = kwargs.pop('fusion_features', ['c1','c2','c3'])
        if self.cfg_pyramid is not None:
            if self.cfg_pyramid['version'] == 'v2':
                self.init_levels = self.cfg_pyramid.get('num_levels', 3)
            else:
                self.init_levels = self.cfg_pyramid.get('num_levels', 4)

        # identify the index of each stage(block)
        # NOTE: As posec3d, no fusion in the final stage
        self.block_idx = [0, 3, 6, 12, 15]  #block outputs index
        if len(self.fusion_features) == 2:
            self.fuse_idx = [3,6]
            inoutchannels = [(192,192), (480,480)]
        elif len(self.fusion_features) == 3 and self.fusion_features == ['c1','c2','c3']:
            self.fuse_idx = [0,3,6]
            inoutchannels = [(64,64), (192,192), (480,480)]
        elif len(self.fusion_features) == 3 and self.fusion_features == ['c2','c3','c4']:
            self.fuse_idx = [3,6,12]
            inoutchannels = [(192,192), (480,480), (832,832)]
        elif len(self.fusion_features) == 4:
            self.fuse_idx = [0,3,6,12]
            inoutchannels = [(64,64), (192,192), (480,480), (832,832)]
        elif len(self.fusion_features) == 5:
            self.fuse_idx = [0,3,6,12,15]
            inoutchannels = [(64,64), (192,192), (480,480), (832,832), (1024,1024)]
        else:
            raise ValueError

        if flag_lateral[0]:
            # pose2rgb
            self.rgb_stream_lateral = nn.ModuleList([Lateral_Conn(inoutchannels[i][0], inoutchannels[i][1], lateral_ksize, lateral_ratio, 'pose2rgb', 
                                                    lateral_variant[0], lateral_interpolate, adapt_first=True if 'c5' in self.fusion_features and i==len(inoutchannels)-1 else False)\
                                                    for i in range(len(inoutchannels))])
        if flag_lateral[1]:
            # rgb2pose
            self.pose_stream_lateral = nn.ModuleList([Lateral_Conn(inoutchannels[i][0], inoutchannels[i][1], lateral_ksize, lateral_ratio, 'rgb2pose', 
                                                    lateral_variant[1], lateral_interpolate, adapt_first=True if 'c5' in self.fusion_features and i==len(inoutchannels)-1 else False)\
                                                    for i in range(len(inoutchannels))])


    def forward(self, x_rgb, x_pose):
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

            if i in self.block_idx[:self.use_block]:
                rgb_fea_lst.append(x_rgb)
                pose_fea_lst.append(x_pose)

        rgb_fused = pose_fused = None
        diff = 1

        for i in range(len(rgb_fea_lst)):
            H, W = rgb_fea_lst[i].shape[-2:]
            try:
                rgb_fea_lst[i] = F.avg_pool3d(rgb_fea_lst[i], (2, H, W), stride=1)  #spatial global average pool
            except:
                rgb_fea_lst[i] = F.avg_pool3d(rgb_fea_lst[i], (1, H, W), stride=1)  #spatial global average pool
            B, C, T = rgb_fea_lst[i].shape[:3]
            rgb_fea_lst[i] = rgb_fea_lst[i].view(B, C, T).permute(0, 2, 1)  #B,T,C
            H, W = pose_fea_lst[i].shape[-2:]
            try:
                pose_fea_lst[i] = F.avg_pool3d(pose_fea_lst[i], (2, H, W), stride=1)  #spatial global average pool
            except:
                pose_fea_lst[i] = F.avg_pool3d(pose_fea_lst[i], (1, H, W), stride=1)  #spatial global average pool
            B, C, T = pose_fea_lst[i].shape[:3]
            pose_fea_lst[i] = pose_fea_lst[i].view(B, C, T).permute(0, 2, 1)  #B,T,C

        return {'rgb_fused': rgb_fused, 'pose_fused': pose_fused, 
                'rgb_fea_lst': rgb_fea_lst[diff:], 'pose_fea_lst': pose_fea_lst[diff:]
                }
