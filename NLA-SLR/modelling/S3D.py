import torch, glob, os
import torch.nn as nn
from utils.misc import get_logger, neq_load_customized
from modelling.S3D_base import S3D_base, BasicConv3d

BLOCK2SIZE = {1:64, 2:192, 3:480, 4:832, 5:1024}
class S3Ds(S3D_base):
    def __init__(self, in_channel=3, use_block=5, freeze_block=0, coord_conv=None):
        self.use_block = use_block
        super(S3Ds, self).__init__(in_channels=in_channel, use_block=use_block, coord_conv=coord_conv)
        self.freeze_block = freeze_block
        self.END_POINT2BLOCK = {
            0: 'block1',
            3: 'block2',
            6: 'block3',
            12: 'block4',
            15: 'block5',
        }
        self.BLOCK2END_POINT = {blk:ep for ep, blk in self.END_POINT2BLOCK.items()}

        self.frozen_modules = []
        self.use_block = use_block

        if freeze_block>0:
            for i in range(0, self.base_num_layers): #base  0,1,2,...,self.BLOCK2END_POINT[blk]
                module_name = 'base.{}'.format(i)
                submodule = self.base[i]
                assert submodule != None, module_name
                if i <= self.BLOCK2END_POINT['block{}'.format(freeze_block)]:
                    self.frozen_modules.append(submodule)


    def forward(self, x):
        x = self.base(x)
        return x


class S3D_backbone(torch.nn.Module):
    def __init__(self, in_channel=3, use_block=5, freeze_block=0, pretrained_ckpt='../../pretrained_models/s3ds_actioncls_ckpt', 
                        cfg_pyramid=None, coord_conv=None, use_shortcut=False):
        super(S3D_backbone, self).__init__()
        self.logger = get_logger()
        self.cfg_pyramid = cfg_pyramid
        self.backbone = S3Ds(in_channel=in_channel, use_block=use_block, freeze_block=freeze_block, coord_conv=coord_conv)  
        self.set_frozen_layers()
        self.out_features = BLOCK2SIZE[use_block]
        if pretrained_ckpt=='scratch':
            self.logger.info('Train S3D backbone from scratch')
        else:
            self.logger.info('Load pretrained S3D backbone from {}'.format(pretrained_ckpt))
            self.load_s3d_model_weight(pretrained_ckpt)
        
        self.stage_idx = [0, 3, 6, 12, 15]
        self.stage_idx = self.stage_idx[:use_block]
        self.use_block = use_block

        self.use_shortcut = use_shortcut
        if use_shortcut:
            dims = [64, 192, 480, 832, 1024]
            k_sizes = [(1,3,3), (1,3,3), (3,3,3), (3,3,3)]
            strides = [(1,2,2), (1,2,2), (2,2,2), (2,2,2)]
            paddings = [(0,1,1), (0,1,1), (1,1,1), (1,1,1)]
            self.shortcut_lst = nn.ModuleList()
            for i in range(use_block-1):
                self.shortcut_lst.append(BasicConv3d(dims[i], dims[i+1], k_sizes[i], strides[i], paddings[i]))
        
        self.pyramid = None
        self.num_levels = 3

    def load_s3d_model_weight(self, model_path):
        if 'actioncls' in model_path:
            filename = glob.glob(os.path.join(model_path, '*.pt'))
            checkpoint = torch.load(filename[0], map_location='cpu')
            state_dict = checkpoint
            new_dict = {}
            for k,v in state_dict.items():
                k = k.replace('module.', 'backbone.')
                new_dict[k] = v
            state_dict = new_dict
            try: self.load_state_dict(state_dict)
            except: neq_load_customized(self, state_dict, verbose=True)
        elif 'glosscls' in model_path:
            filename = glob.glob(os.path.join(model_path, '*.pth.tar'))
            checkpoint = torch.load(filename[0], map_location='cpu')
            state_dict = checkpoint['state_dict']
            try:
                self.load_state_dict(state_dict)
            except:
                neq_load_customized(self, state_dict, verbose=True)
        else:
            raise ValueError  

    def set_train(self):
        self.train()
        for m in getattr(self.backbone,'frozen_modules',[]):
            m.eval()

    def get_frozen_layers(self):
        return getattr(self.backbone,'frozen_modules',[])

    def set_frozen_layers(self):
        for m in getattr(self.backbone,'frozen_modules',[]):
            for param in m.parameters():
                #print(param)
                param.requires_grad = False
            m.eval()

    def forward(self, sgn_videos):
        (B, C, T_in, H, W) = sgn_videos.shape

        # feat3d = self.backbone(sgn_videos) 
        fea_lst = []
        shortcut_fea_lst = []
        for i, layer in enumerate(self.backbone.base):
            sgn_videos = layer(sgn_videos)

            # shortcut
            if self.use_shortcut and i in self.stage_idx:
                if i in self.stage_idx[1:]:
                    sgn_videos = sgn_videos + self.shortcut_lst[self.stage_idx.index(i)-1](shortcut_fea_lst[-1])
                shortcut_fea_lst.append(sgn_videos)

            if i in self.stage_idx[self.use_block-self.num_levels:]:
                # first block is too shallow, drop it
                fea_lst.append(sgn_videos)
                # print(sgn_videos.shape)
        
        if self.pyramid is not None:
            fea_lst, _ = self.pyramid(fea_lst)
        
        return {'sgn_feature':fea_lst[-1], 'fea_lst': fea_lst}