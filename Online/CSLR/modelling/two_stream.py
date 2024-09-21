import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from modelling.fusion import Lateral_Conn
from modelling.S3D import S3D_backbone
from modelling.pyramid import PyramidNetwork


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
        
        dpr = [x.item() for x in torch.linspace(0, kwargs.pop('drop_path_rate', 0.), self.use_block-1)]
        self.drop_path_lst = nn.ModuleList()
        for d in dpr:
            self.drop_path_lst.append(DropPath(d) if d > 0. else nn.Identity())

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

        self.word_emb_tab = kwargs.pop('word_emb_tab', None)
        if self.word_emb_tab is not None:
            self.word_emb_dim = kwargs.pop('word_emb_dim', 0)
            if self.word_emb_tab is not None:
                self.word_emb_tab.requires_grad = False
                self.word_emb_mapper_rgb = nn.ModuleList()
                self.word_emb_mapper_keypoint = nn.ModuleList()
                for c in inoutchannels:
                    self.word_emb_mapper_rgb.append(nn.Linear(self.word_emb_dim, c[0]))
                    self.word_emb_mapper_keypoint.append(nn.Linear(self.word_emb_dim, c[0]))
            self.contras_setting = kwargs.pop('contras_setting', 'frame')
            if 'extrafc' in self.contras_setting:
                self.linear_rgb = nn.ModuleList()
                self.linear_keypoint = nn.ModuleList()
                for c in inoutchannels:
                    self.linear_rgb.append(nn.Linear(c[0], c[0]))
                    self.linear_keypoint.append(nn.Linear(c[0], c[0]))
            self.temp = kwargs.pop('temp', 0.1)


    def word_emb_att(self, x, idx, stream='rgb'):
        # B,C,T,H,W = x.shape
        if stream == 'rgb':
            k = self.word_emb_mapper_rgb[idx](self.word_emb_tab)  #[N,C]
        elif stream == 'keypoint':
            k = self.word_emb_mapper_keypoint[idx](self.word_emb_tab)  #[N,C]
        q = x.permute(0,2,3,4,1)
        if 'extrafc' in self.contras_setting:
            if stream == 'rgb':
                q = self.linear_rgb[idx](q)
            elif stream == 'keypoint':
                q = self.linear_keypoint[idx](q)
        if 'video' in self.contras_setting:
            q = q.mean(dim=(1,2,3))
        norm_q = F.normalize(q, dim=-1)
        norm_k = F.normalize(k, dim=-1)
        if type(self.temp) == float:
            scores = torch.matmul(norm_q, norm_k.transpose(1,0)) / self.temp  #[B,N]
        else:
            scores = torch.matmul(norm_q, norm_k.transpose(1,0)) / torch.sigmoid(self.temp)
        return_scores = scores
        scores = F.softmax(scores, dim=-1)
        contexts = torch.matmul(scores, k)
        if 'video' in self.contras_setting:
            contexts = contexts.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x + contexts
        return x, return_scores


    def forward(self, x_rgb, x_pose):
        B, C, T_in, H, W = x_rgb.shape
        rgb_fea_lst, pose_fea_lst = [], []
        rgb_score_lst, pose_score_lst = [], []
        rgb_shortcut_lst, pose_shortcut_lst = [], []
        for i, (rgb_layer, pose_layer) in enumerate(zip(self.rgb_stream.backbone.base, self.pose_stream.backbone.base)):
            x_rgb = rgb_layer(x_rgb)
            x_pose = pose_layer(x_pose)

            if self.use_shortcut and i in self.block_idx:
                if i in self.block_idx[1:]:
                    x_rgb = self.drop_path_lst[self.block_idx.index(i)-1](x_rgb) + self.rgb_stream.shortcut_lst[self.block_idx.index(i)-1](rgb_shortcut_lst[-1])
                    x_pose = self.drop_path_lst[self.block_idx.index(i)-1](x_pose) + self.pose_stream.shortcut_lst[self.block_idx.index(i)-1](pose_shortcut_lst[-1])
                rgb_shortcut_lst.append(x_rgb)
                pose_shortcut_lst.append(x_pose)

            if i in self.fuse_idx:
                x_rgb_fused = x_rgb
                x_pose_fused = x_pose

                if self.flag_lateral[0] and 'p' not in self.fusion_features[self.fuse_idx.index(i)]:
                    # print(self.fusion_features[self.fuse_idx.index(i)])
                    _, x_rgb_fused = self.rgb_stream_lateral[self.fuse_idx.index(i)](x_rgb, x_pose)
                if self.flag_lateral[1] and 'p' not in self.fusion_features[self.fuse_idx.index(i)]:
                    _, x_pose_fused = self.pose_stream_lateral[self.fuse_idx.index(i)](x_rgb, x_pose)
                
                if self.word_emb_tab is not None:
                    x_rgb_fused, rgb_scores = self.word_emb_att(x_rgb_fused, idx=self.fuse_idx.index(i), stream='rgb')
                    x_pose_fused, pose_scores = self.word_emb_att(x_pose_fused, idx=self.fuse_idx.index(i), stream='keypoint')
                    rgb_score_lst.append(rgb_scores)
                    pose_score_lst.append(pose_scores)

                x_rgb = x_rgb_fused
                x_pose = x_pose_fused

            if i in self.block_idx[:self.use_block]:
                rgb_fea_lst.append(x_rgb)
                pose_fea_lst.append(x_pose)

        rgb_fused = pose_fused = None
        diff = 1
        if self.cfg_pyramid['rgb'] or self.cfg_pyramid['pose']:
            tot = len(rgb_fea_lst)
            diff = tot - self.init_levels
            for i in range(tot-1, diff, -1):
                if self.rgb_stream.pyramid:
                    if self.cfg_pyramid['version'] == 'v2':
                        rgb_fea_lst[i-1] = rgb_fea_lst[i-1] + self.rgb_stream.pyramid.upsample_layers[tot-i-1](rgb_fea_lst[i])
                        rgb_fea_lst[i-1] = self.rgb_stream.pyramid.conv1x1_layers[tot-i-1](rgb_fea_lst[i-1])
                    else:
                        rgb_fea_lst[i-1] = rgb_fea_lst[i-1] + self.rgb_stream.pyramid.upsample_layers[tot-i-1](rgb_fea_lst[i])

                if self.pose_stream.pyramid:
                    if self.cfg_pyramid['version'] == 'v2':
                        pose_fea_lst[i-1] = pose_fea_lst[i-1] + self.pose_stream.pyramid.upsample_layers[tot-i-1](pose_fea_lst[i])
                        pose_fea_lst[i-1] = self.pose_stream.pyramid.conv1x1_layers[tot-i-1](pose_fea_lst[i-1])
                    else:
                        pose_fea_lst[i-1] = pose_fea_lst[i-1] + self.pose_stream.pyramid.upsample_layers[tot-i-1](pose_fea_lst[i])

                if 'p'+str(i) in self.fusion_features:
                    # print('p'+str(i))
                    _, rgb_fea_lst[i-1] = self.rgb_stream_lateral[self.fusion_features.index('p'+str(i))](rgb_fea_lst[i-1], pose_fea_lst[i-1])
                    _, pose_fea_lst[i-1] = self.pose_stream_lateral[self.fusion_features.index('p'+str(i))](rgb_fea_lst[i-1], pose_fea_lst[i-1])

        for i in range(len(rgb_fea_lst)):
            H, W = rgb_fea_lst[i].shape[-2:]
            if i < len(rgb_fea_lst)-1:
                rgb_fea_lst[i] = F.avg_pool3d(rgb_fea_lst[i], (2, H, W), stride=1)  #spatial global average pool
            else:
                rgb_fea_lst[i] = F.avg_pool3d(rgb_fea_lst[i], (1, H, W), stride=1)  #spatial global average pool
            B, C, T = rgb_fea_lst[i].shape[:3]
            rgb_fea_lst[i] = rgb_fea_lst[i].view(B, C, T).permute(0, 2, 1)  #B,T,C
            H, W = pose_fea_lst[i].shape[-2:]
            if i < len(rgb_fea_lst)-1:
                pose_fea_lst[i] = F.avg_pool3d(pose_fea_lst[i], (2, H, W), stride=1)  #spatial global average pool
            else:
                pose_fea_lst[i] = F.avg_pool3d(pose_fea_lst[i], (1, H, W), stride=1)  #spatial global average pool
            B, C, T = pose_fea_lst[i].shape[:3]
            pose_fea_lst[i] = pose_fea_lst[i].view(B, C, T).permute(0, 2, 1)  #B,T,C
            

        # rgb_out = pose_out = None
        # if self.cfg_pyramid['rgb'] is None:
        #     rgb_out = rgb_fea_lst[-1]
        # if self.cfg_pyramid['pose'] is None:
        #     pose_out = pose_fea_lst[-1]

        return {'rgb_fused': rgb_fused, 'pose_fused': pose_fused, 
                'rgb_fea_lst': rgb_fea_lst[diff:], 'pose_fea_lst': pose_fea_lst[diff:],
                'rgb_word_emb_score_lst': rgb_score_lst, 'pose_word_emb_score_lst': pose_score_lst}



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