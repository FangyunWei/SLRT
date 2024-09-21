import torch
from torch import nn
import torch.nn.functional as F
from modelling.cbam import CBAM_twostream


class Lateral_Conn(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size=(7,3,3), ratio=(1,2,2), direction='rgb2pose', variant=None):
        super(Lateral_Conn, self).__init__()
        # ratio: temporal, height, width
        # kernel_size: temporal, height, width. 7 for temporal, same as posec3d. 3 for spatial, intuitively set
        # if bidirection, then create two fuser
        assert direction in ['rgb2pose', 'pose2rgb']
        self.direction = direction

        self.variant = variant
        if variant == 'spat_att':
            # only keep spatial attention module
            self.spat_att_mod = CBAM_twostream(inchannels, channel_spatial=[False, True])

        padding = (kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2)
        if self.direction == 'rgb2pose':
            self.conv = nn.Conv3d(inchannels, outchannels, kernel_size, ratio, padding, bias=False)
        elif self.direction == 'pose2rgb':
            output_padding = (0, 1, 1)
            self.conv = nn.ConvTranspose3d(inchannels, outchannels, kernel_size, ratio, padding, output_padding, bias=False)
        self.init_weights()
    

    def forward(self, x_rgb, x_pose):
        if self.direction == 'rgb2pose':
            x_rgb = self.conv(x_rgb)
        elif self.direction == 'pose2rgb':
            x_pose = self.conv(x_pose)
        
        if self.variant == 'spat_att':
            spat_att, x_fused = self.spat_att_mod(x_rgb, x_pose)
            return spat_att, x_fused
        else:
            return None, x_rgb+x_pose
    

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight)


# if __name__ == '__main__':
#     x_rgb = torch.rand(2,256,10,28,28).cuda()
#     x_pose = torch.rand(2,256,10,14,14).cuda()
#     model = Lateral_Conn(256, 256, (7,3,3), (1,2,2), 'pose2rgb').cuda()
#     res = model(x_rgb, x_pose)
#     print(res.shape)