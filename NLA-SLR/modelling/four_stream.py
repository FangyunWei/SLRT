import torch
from torch import nn
import torch.nn.functional as F
from modelling.two_stream import S3D_two_stream_v2
from modelling.fusion import Lateral_Conn


class S3D_four_stream(nn.Module):
    def __init__(self, use_block=5, freeze_block=(0,0), pose_inchannels=17, flag_lateral=[True, True], **kwargs):
        super(S3D_four_stream, self).__init__()
        self.ts_high = S3D_two_stream_v2(use_block, freeze_block, pose_inchannels, flag_lateral, **kwargs)  #high frame numbers
        self.ts_low = S3D_two_stream_v2(use_block, freeze_block, pose_inchannels, flag_lateral, **kwargs)   #low frame numbers

        inoutchannels = [(64,64), (192,192), (480,480), (832,832)]
        if flag_lateral[2]:
            self.rgb_low2high = nn.ModuleList([Lateral_Conn(inoutchannels[i][0], inoutchannels[i][1], (3,1,1), (2,1,1), 'pose2rgb', 
                                                None, False, adapt_first=False) for i in range(len(inoutchannels))])
        if flag_lateral[3]:
            self.rgb_high2low = nn.ModuleList([Lateral_Conn(inoutchannels[i][0], inoutchannels[i][1], (3,1,1), (2,1,1), 'rgb2pose', 
                                                None, False, adapt_first=False) for i in range(len(inoutchannels))])
        if flag_lateral[4]:
            self.pose_low2high = nn.ModuleList([Lateral_Conn(inoutchannels[i][0], inoutchannels[i][1], (3,1,1), (2,1,1), 'pose2rgb', 
                                                None, False, adapt_first=False) for i in range(len(inoutchannels))])
        if flag_lateral[5]:
            self.pose_high2low = nn.ModuleList([Lateral_Conn(inoutchannels[i][0], inoutchannels[i][1], (3,1,1), (2,1,1), 'rgb2pose', 
                                                None, False, adapt_first=False) for i in range(len(inoutchannels))])
        self.flag_lateral = flag_lateral
    

    def forward(self, x_rgb_high, x_rgb_low, x_pose_high, x_pose_low):
        B, C, T_in, H, W = x_rgb_high.shape
        for i, (rgb_high_layer, pose_high_layer, rgb_low_layer, pose_low_layer) in enumerate(zip(self.ts_high.rgb_stream.backbone.base, self.ts_high.pose_stream.backbone.base, 
            self.ts_low.rgb_stream.backbone.base, self.ts_low.pose_stream.backbone.base)):

            x_rgb_high = rgb_high_layer(x_rgb_high)
            x_pose_high = pose_high_layer(x_pose_high)
            x_rgb_low = rgb_low_layer(x_rgb_low)
            x_pose_low = pose_low_layer(x_pose_low)

            if i in self.ts_high.fuse_idx:
                # print(i)
                # high, intra ts
                if self.flag_lateral[0] and self.flag_lateral[1]:
                    _, x_rgb_high_fused = self.ts_high.rgb_stream_lateral[self.ts_high.fuse_idx.index(i)](x_rgb_high, x_pose_high)
                    _, x_pose_high_fused = self.ts_high.pose_stream_lateral[self.ts_high.fuse_idx.index(i)](x_rgb_high, x_pose_high)
                    x_rgb_high = x_rgb_high_fused
                    x_pose_high = x_pose_high_fused

                # low, intra ts
                if self.flag_lateral[0] and self.flag_lateral[1]:
                    _, x_rgb_low_fused = self.ts_low.rgb_stream_lateral[self.ts_low.fuse_idx.index(i)](x_rgb_low, x_pose_low)
                    _, x_pose_low_fused = self.ts_low.pose_stream_lateral[self.ts_low.fuse_idx.index(i)](x_rgb_low, x_pose_low)
                    x_rgb_low = x_rgb_low_fused
                    x_pose_low = x_pose_low_fused

                # RGB, inter ts
                if self.flag_lateral[2] and self.flag_lateral[3]:
                    _, x_rgb_low_fused = self.rgb_high2low[self.ts_high.fuse_idx.index(i)](x_rgb_high, x_rgb_low)
                    _, x_rgb_high_fused = self.rgb_low2high[self.ts_high.fuse_idx.index(i)](x_rgb_high, x_rgb_low)
                    x_rgb_low = x_rgb_low_fused
                    x_rgb_high = x_rgb_high_fused

                # Pose, inter ts
                if self.flag_lateral[4] and self.flag_lateral[5]:
                    _, x_pose_low_fused = self.pose_high2low[self.ts_high.fuse_idx.index(i)](x_pose_high, x_pose_low)
                    _, x_pose_high_fused = self.pose_low2high[self.ts_high.fuse_idx.index(i)](x_pose_high, x_pose_low)
                    x_pose_low = x_pose_low_fused
                    x_pose_high = x_pose_high_fused

            # if i in self.block_idx[:self.use_block]:
            #     rgb_fea_lst.append(x_rgb)
            #     pose_fea_lst.append(x_pose)

        H, W = x_rgb_high.shape[-2:]
        try:
            x_rgb_high = F.avg_pool3d(x_rgb_high, (2, H, W), stride=1)  #spatial global average pool
        except:
            x_rgb_high = F.avg_pool3d(x_rgb_high, (1, H, W), stride=1)
        try:
            x_rgb_low = F.avg_pool3d(x_rgb_low, (2, H, W), stride=1)
        except:
            x_rgb_low = F.avg_pool3d(x_rgb_low, (1, H, W), stride=1)
        B, C, T = x_rgb_high.shape[:3]
        x_rgb_high = x_rgb_high.view(B, C, T).permute(0, 2, 1).mean(dim=1)  #B,C
        B, C, T = x_rgb_low.shape[:3]
        x_rgb_low = x_rgb_low.view(B, C, T).permute(0, 2, 1).mean(dim=1)  #B,C

        H, W = x_pose_high.shape[-2:]
        try:
            x_pose_high = F.avg_pool3d(x_pose_high, (2, H, W), stride=1)  #spatial global average pool
        except:
            x_pose_high = F.avg_pool3d(x_pose_high, (1, H, W), stride=1)
        try:
            x_pose_low = F.avg_pool3d(x_pose_low, (2, H, W), stride=1)
        except:
            x_pose_low = F.avg_pool3d(x_pose_low, (1, H, W), stride=1)
        B, C, T = x_pose_high.shape[:3]
        x_pose_high = x_pose_high.view(B, C, T).permute(0, 2, 1).mean(dim=1)  #B,C
        B, C, T = x_pose_low.shape[:3]
        x_pose_low = x_pose_low.view(B, C, T).permute(0, 2, 1).mean(dim=1)  #B,C

        return {'rgb-h': x_rgb_high, 'rgb-l': x_rgb_low, 'kp-h': x_pose_high, 'kp-l': x_pose_low}