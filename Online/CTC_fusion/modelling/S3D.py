import torch, glob, os
from utils.misc import get_logger, neq_load_customized
from modelling.models_3d.S3D.model import S3Dsup
from modelling.pyramid import PyramidNetwork, PyramidNetwork_v2

BLOCK2SIZE = {1:64, 2:192, 3:480, 4:832, 5:1024}
class S3Ds(S3Dsup):
    def __init__(self, in_channel=3, use_block=5, freeze_block=0, stride=2):
        self.use_block = use_block
        super(S3Ds, self).__init__(in_channels=in_channel, num_class=400, use_block=use_block, stride=stride)
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
    def __init__(self, 
        in_channel=3,
        use_block=5, freeze_block=0, stride=2, pretrained_ckpt='scratch', cfg_pyramid=None):
        super(S3D_backbone, self).__init__()
        self.logger = get_logger()
        self.cfg_pyramid = cfg_pyramid
        self.backbone = S3Ds(
            in_channel=in_channel,
            use_block=use_block, freeze_block=freeze_block, stride=stride)  
        self.set_frozen_layers()
        self.out_features = BLOCK2SIZE[use_block]
        if pretrained_ckpt=='scratch':
            self.logger.info('Train S3D backbone from scratch')
        else:
            self.logger.info('Load pretrained S3D backbone from {}'.format(pretrained_ckpt))
            self.load_s3d_model_weight(pretrained_ckpt)
        
        self.stage_idx = [0, 3, 6, 12]
        self.stage_idx = self.stage_idx[:use_block]
        self.use_block = use_block

        if in_channel == 3:
            branch = 'rgb'
        else:
            branch = 'pose'
        
        self.pyramid = None
        self.num_levels = 3
        if cfg_pyramid is not None:
            if cfg_pyramid[branch]:
                if cfg_pyramid['version'] == 'v2':
                    self.num_levels = cfg_pyramid.get('num_levels', 3)
                    self.pyramid = PyramidNetwork_v2(channels=[832,480,192,64], kernel_size=1, num_levels=self.num_levels, temp_scale=[2,1,1], spat_scale=[2,2,2])
                else:
                    self.num_levels = cfg_pyramid.get('num_levels', 4)
                    self.pyramid = PyramidNetwork(channels=[832,480,192,64], kernel_size=3, num_levels=self.num_levels, temp_scale=[2,1,1], spat_scale=[2,2,2])

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

    def forward(self, sgn_videos, sgn_lengths=None):
        (B, C, T_in, H, W) = sgn_videos.shape

        # feat3d = self.backbone(sgn_videos) 
        fea_lst = []
        for i, layer in enumerate(self.backbone.base):
            sgn_videos = layer(sgn_videos)
            if i in self.stage_idx[self.use_block-self.num_levels:]:
                # first block is too shallow, drop it
                fea_lst.append(sgn_videos)
        
        sgn_mask_lst, valid_len_out_lst = [], []
        if self.pyramid is not None:
            fea_lst, _ = self.pyramid(fea_lst)
            for i in range(len(fea_lst)):
                B, T_out, _ = fea_lst[i].shape
                sgn_mask = torch.zeros([B,1,T_out], dtype=torch.bool, device=fea_lst[i].device)
                valid_len_out = torch.floor(sgn_lengths*T_out/T_in).long() #B,
                for bi in range(B):
                    sgn_mask[bi, :, :valid_len_out[bi]] = True
                sgn_mask_lst.append(sgn_mask)
                valid_len_out_lst.append(valid_len_out)
            return {'sgn_feature':fea_lst[-1], 'sgn_mask':sgn_mask_lst, 'valid_len_out': valid_len_out_lst, 'fea_lst': fea_lst}

        else: 
            feat3d = fea_lst[-1]
            B, _, T_out, _, _ = feat3d.shape
            pooled_sgn_feature = torch.mean(feat3d, dim=[3,4]) #B, D, T_out
            sgn = torch.transpose(pooled_sgn_feature, 1, 2) #b, t_OUT, d
            sgn_mask = torch.zeros([B,1,T_out], dtype=torch.bool, device=sgn.device)
            valid_len_out = torch.floor(sgn_lengths*T_out/T_in).long() #B,
            for bi in range(B):
                sgn_mask[bi, :, :valid_len_out[bi]] = True
            sgn_mask_lst.append(sgn_mask)
            valid_len_out_lst.append(valid_len_out)
        
            return {'sgn_feature':fea_lst[-1], 'sgn_mask':sgn_mask_lst, 
                    'valid_len_out': valid_len_out_lst, 'fea_lst': fea_lst, 'sgn':sgn}