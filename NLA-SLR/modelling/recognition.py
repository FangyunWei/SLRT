import torch
from modelling.S3D import S3D_backbone
from modelling.two_stream import S3D_two_stream_v2
from modelling.four_stream import S3D_four_stream
from utils.misc import get_logger, neq_load_customized
import random, torchvision
from modelling.Visualhead import SepConvVisualHead
import numpy as np
from copy import deepcopy
from utils.gen_gaussian import gen_gaussian_hmap_op
import math
from utils.loss import LabelSmoothCE


class RecognitionNetwork(torch.nn.Module):
    def __init__(self, cfg, transform_cfg, cls_num=2000, input_streams=['rgb'], input_frames=64, word_emb_tab=None):
        super().__init__()
        logger = get_logger()
        self.cfg = cfg
        self.input_streams = input_streams
        self.fuse_method = cfg.get('fuse_method', None)
        self.heatmap_cfg = cfg.get('heatmap_cfg', {})
        self.traj_hmap_cfg = cfg.get('traj_hmap_cfg', {})
        self.transform_cfg = transform_cfg
        self.preprocess_chunksize = self.heatmap_cfg.get('preprocess_chunksize', 16)
        self.word_emb_tab = word_emb_tab
        
        cfg['pyramid'] = cfg.get('pyramid', {'version':None, 'rgb':None, 'pose':None})
        self.visual_backbone = self.visual_backbone_keypoint = self.visual_backbone_twostream = None
        if input_streams == ['rgb']:
            self.visual_backbone = S3D_backbone(in_channel=3, **cfg['s3d'], cfg_pyramid=cfg['pyramid'])
            
        elif input_streams == ['keypoint']:
            self.visual_backbone_keypoint = S3D_backbone(**cfg['keypoint_s3d'], cfg_pyramid=cfg['pyramid'])
            
        elif len(input_streams) == 2:
            self.visual_backbone_twostream = S3D_two_stream_v2(
                use_block=cfg['s3d']['use_block'],
                freeze_block=(cfg['s3d']['freeze_block'], cfg['keypoint_s3d']['freeze_block']),
                pose_inchannels=cfg['keypoint_s3d']['in_channel'],
                flag_lateral=(cfg['lateral'].get('pose2rgb',False), cfg['lateral'].get('rgb2pose',False)),
                lateral_variant=(cfg['lateral'].get('variant_pose2rgb', None), cfg['lateral'].get('variant_rgb2pose', None)),
                lateral_ksize=tuple(cfg['lateral'].get('kernel_size', (1,3,3))),
                lateral_ratio=tuple(cfg['lateral'].get('ratio', (1,2,2))),
                lateral_interpolate=cfg['lateral'].get('interpolate', False),
                cfg_pyramid=cfg['pyramid'],
                fusion_features=cfg['lateral'].get('fusion_features',['c1','c2','c3'])
            )
        
        elif len(input_streams) == 4:
            self.visual_backbone_fourstream = S3D_four_stream(
                use_block=cfg['s3d']['use_block'],
                freeze_block=(cfg['s3d']['freeze_block'], cfg['keypoint_s3d']['freeze_block']),
                pose_inchannels=cfg['keypoint_s3d']['in_channel'],
                flag_lateral=(cfg['lateral'].get('pose2rgb',True), cfg['lateral'].get('rgb2pose',True),
                                cfg['lateral'].get('rgb_low2high',True), cfg['lateral'].get('rgb_high2low',True),
                                cfg['lateral'].get('pose_low2high',True), cfg['lateral'].get('pose_high2low',True)),
                lateral_variant=(cfg['lateral'].get('variant_pose2rgb', None), cfg['lateral'].get('variant_rgb2pose', None)),
                lateral_ksize=tuple(cfg['lateral'].get('kernel_size', (1,3,3))),
                lateral_ratio=tuple(cfg['lateral'].get('ratio', (1,2,2))),
                lateral_interpolate=cfg['lateral'].get('interpolate', False),
                cfg_pyramid=cfg['pyramid'],
                fusion_features=cfg['lateral'].get('fusion_features',['c1','c2','c3'])
            )

        if 'visual_head' in cfg:
            HeadCLS = SepConvVisualHead
            language_apply_to = cfg.get('language_apply_to', 'rgb_keypoint_joint')
            if 'rgb' in input_streams or len(self.input_streams)==2:
                rgb_head_cfg = deepcopy(cfg['visual_head'])
                if 'rgb' not in language_apply_to:
                    rgb_head_cfg['contras_setting'] = None
                self.visual_head = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **rgb_head_cfg)
            else:
                self.visual_head = None
            
            if 'keypoint' in input_streams or 'keypoint_coord' in input_streams or len(self.input_streams)==2:
                keypoint_head_cfg = deepcopy(cfg['visual_head'])
                if 'keypoint' not in language_apply_to:
                    keypoint_head_cfg['contras_setting'] = None
                self.visual_head_keypoint = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **keypoint_head_cfg)
            else:
                self.visual_head_keypoint = None
            
            if len(input_streams) == 4:
                self.visual_head = self.visual_head_keypoint = None
                self.visual_head_rgb_h = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **rgb_head_cfg)
                self.visual_head_rgb_l = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **rgb_head_cfg)
                self.visual_head_kp_h = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **keypoint_head_cfg)
                self.visual_head_kp_l = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **keypoint_head_cfg)
                self.head_dict = {'rgb-h': self.visual_head_rgb_h, 'rgb-l': self.visual_head_rgb_l,
                                'kp-h': self.visual_head_kp_h, 'kp-l': self.visual_head_kp_l,
                                'fuse': None, 'fuse-h': None, 'fuse-l': None, 'fuse-x-rgb': None, 'fuse-x-kp': None}
            
            if self.fuse_method is not None and 'triplehead' in self.fuse_method:
                assert len(input_streams)==2
                joint_head_cfg = deepcopy(cfg['visual_head'])
                if 'joint' not in language_apply_to:
                    joint_head_cfg['contras_setting'] = None
                if 'cat' in self.fuse_method:
                    joint_head_cfg['input_size'] = 2*cfg['visual_head']['input_size'] #dirty solution
                self.visual_head_fuse = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **joint_head_cfg)

            elif self.fuse_method is not None and 'four' in self.fuse_method:
                assert len(input_streams)==4
                joint_head_cfg = deepcopy(cfg['visual_head'])
                if 'joint' not in language_apply_to:
                    joint_head_cfg['contras_setting'] = None
                if 'catall' in self.fuse_method or 'type3' in self.fuse_method:
                    joint_head_cfg['input_size'] = 4*cfg['visual_head']['input_size']
                    self.visual_head_fuse = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **joint_head_cfg)
                    self.head_dict['fuse'] = self.visual_head_fuse
                if 'type' in self.fuse_method:
                    joint_head_cfg['input_size'] = 2*cfg['visual_head']['input_size']
                    self.visual_head_fuse_h = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **joint_head_cfg)
                    self.visual_head_fuse_l = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **joint_head_cfg)
                    self.head_dict['fuse-h'] = self.visual_head_fuse_h
                    self.head_dict['fuse-l'] = self.visual_head_fuse_l
                    if 'type2' in self.fuse_method:
                        self.visual_head_x_rgb = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **joint_head_cfg)
                        self.visual_head_x_kp = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **joint_head_cfg)
                        self.head_dict['fuse-x-rgb'] = self.visual_head_x_rgb
                        self.head_dict['fuse-x-kp'] = self.visual_head_x_kp

        if 'pretrained_path_rgb' in cfg:
            load_dict = torch.load(cfg['pretrained_path_rgb'],map_location='cpu')['model_state']      
            backbone_dict, head_dict, head_remain_dict = {}, {}, {}
            for k, v in load_dict.items():
                if 'visual_backbone' in k:
                    backbone_dict[k.replace('recognition_network.visual_backbone.','')] = v
                if 'visual_head' in k and 'visual_head_remain' not in k:
                    head_dict[k.replace('recognition_network.visual_head.','')] = v
            if self.visual_backbone!=None and self.visual_backbone_twostream==None:
                neq_load_customized(self.visual_backbone, backbone_dict, verbose=True)
                neq_load_customized(self.visual_head, head_dict, verbose=True)
                logger.info('Load visual_backbone and visual_head for rgb from {}'.format(cfg['pretrained_path_rgb']))
            elif self.visual_backbone==None and self.visual_backbone_twostream!=None:
                neq_load_customized(self.visual_backbone_twostream.rgb_stream, backbone_dict, verbose=True)
                neq_load_customized(self.visual_head, head_dict, verbose=True)
                logger.info('Load visual_backbone_twostream.rgb_stream and visual_head for rgb from {}'.format(cfg['pretrained_path_rgb'])) 
            else:
                logger.info('No rgb stream exists in the network')

        if 'pretrained_path_keypoint' in cfg and input_streams != ['keypoint_coord']:
            load_dict = torch.load(cfg['pretrained_path_keypoint'], map_location='cpu')['model_state']
            backbone_dict, head_dict, head_remain_dict = {}, {}, {}
            for k, v in load_dict.items():
                if 'visual_backbone_keypoint' in k:
                    backbone_dict[k.replace('recognition_network.visual_backbone_keypoint.','')] = v
                if 'visual_head_keypoint' in k and 'visual_head_keypoint_remain' not in k: #for model trained using new_code
                    head_dict[k.replace('recognition_network.visual_head_keypoint.','')] = v
                elif 'visual_head_keypoint_remain' in k:
                    head_remain_dict[k.replace('recognition_network.visual_head_keypoint_remain.','')] = v
            if self.visual_backbone_keypoint!=None and self.visual_backbone_twostream==None:
                neq_load_customized(self.visual_backbone_keypoint, backbone_dict, verbose=True)
                neq_load_customized(self.visual_head_keypoint, head_dict, verbose=True)
                logger.info('Load visual_backbone and visual_head for keypoint from {}'.format(cfg['pretrained_path_keypoint']))
            elif self.visual_backbone_keypoint==None and self.visual_backbone_twostream!=None:
                neq_load_customized(self.visual_backbone_twostream.pose_stream, backbone_dict, verbose=True)
                neq_load_customized(self.visual_head_keypoint, head_dict, verbose=True)
                logger.info('Load visual_backbone_twostream.pose_stream and visual_head for pose from {}'.format(cfg['pretrained_path_keypoint']))    
            else:
                logger.info('No pose stream exists in the network')
        
        if 'pretrained_path_two' in cfg:
            assert len(input_streams)==4
            load_dict = {}
            load_dict['high'] = torch.load(cfg['pretrained_path_two'][0], map_location='cpu')['model_state']
            load_dict['low'] = torch.load(cfg['pretrained_path_two'][1], map_location='cpu')['model_state']
            backbone_dict, head_dict = {'high': {}, 'low': {}}, {'high': {'rgb': {}, 'keypoint': {}, 'fuse': {}}, 'low': {'rgb': {}, 'keypoint': {}, 'fuse': {}}}
            for n in ['high', 'low']:
                for k, v in load_dict[n].items():
                    if 'visual_backbone_twostream' in k:
                        backbone_dict[n][k.replace('recognition_network.visual_backbone_twostream.','')] = v
                    if 'visual_head_keypoint' in k:
                        head_dict[n]['keypoint'][k.replace('recognition_network.visual_head_keypoint.','')] = v
                    elif 'visual_head_fuse' in k:
                        head_dict[n]['fuse'][k.replace('recognition_network.visual_head_fuse.','')] = v
                    elif 'visual_head' in k:
                        head_dict[n]['rgb'][k.replace('recognition_network.visual_head.','')] = v
            
            neq_load_customized(self.visual_backbone_fourstream.ts_high, backbone_dict['high'])
            logger.info('Load visual_backbone high')
            neq_load_customized(self.visual_backbone_fourstream.ts_low, backbone_dict['low'])
            logger.info('Load visual_backbone low')
            neq_load_customized(self.visual_head_rgb_h, head_dict['high']['rgb'])
            logger.info('Load visual_head rgb high')
            neq_load_customized(self.visual_head_rgb_l, head_dict['low']['rgb'])
            logger.info('Load visual_head rgb low')
            neq_load_customized(self.visual_head_kp_h, head_dict['high']['keypoint'])
            logger.info('Load visual_head keypoint high')
            neq_load_customized(self.visual_head_kp_l, head_dict['low']['keypoint'])
            logger.info('Load visual_head keypoint low')
            if self.head_dict['fuse-h'] is not None:
                neq_load_customized(self.visual_head_fuse_h, head_dict['high']['fuse'])
                logger.info('Load visual_head fuse high')
            if self.head_dict['fuse-l'] is not None:
                neq_load_customized(self.visual_head_fuse_l, head_dict['low']['fuse'])
                logger.info('Load visual_head fuse low')  

        label_smooth = cfg.get('label_smooth', 0.0)
        if type(label_smooth)==float and label_smooth > 0:
            self.recognition_loss_func = LabelSmoothCE(lb_smooth=label_smooth, reduction='mean')
        elif type(label_smooth)==str and 'word_emb_sim' in label_smooth:
            temp, lb_smooth, norm_type = float(label_smooth.split('_')[-1]), float(label_smooth.split('_')[-2]), label_smooth.split('_')[-3]
            variant = 'word_sim'
            self.recognition_loss_func = LabelSmoothCE(lb_smooth=lb_smooth, reduction='mean', 
                        word_emb_tab=word_emb_tab, norm_type=norm_type, temp=temp, variant=variant)
        else:
            self.recognition_loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.contras_setting = cfg['visual_head'].get('contras_setting', None)
        self.contras_loss_weight = cfg.get('contras_loss_weight', 1.0)
        if self.contras_setting and 'dual' in self.contras_setting:
            self.contras_loss_func = LabelSmoothCE(reduction='mean', variant=self.contras_setting)


    def compute_recognition_loss(self, logits, labels, **kwargs):
        if isinstance(self.recognition_loss_func, torch.nn.CrossEntropyLoss):
            return self.recognition_loss_func(logits, labels)
        else:
            return self.recognition_loss_func(logits, labels, **kwargs)


    def decode(self, gloss_logits, k=10):
        # get topk decoded results
        res = torch.argsort(gloss_logits, dim=-1, descending=True)
        res = res[..., :k]
        return res


    def generate_batch_heatmap(self, keypoints, heatmap_cfg):
        #self.sigma
        #keypoints B,T,N,3
        B,T,N,D = keypoints.shape
        keypoints = keypoints.reshape(-1, N, D)
        chunk_size = int(math.ceil((B*T)/self.preprocess_chunksize))
        chunks = torch.split(keypoints, chunk_size, dim=0)

        heatmaps = []
        for chunk in chunks:
            # print(chunk.shape)
            hm = gen_gaussian_hmap_op(
                coords=chunk,  
                **heatmap_cfg) #sigma, confidence, threshold) #B*T,N,H,W
            N, H, W = hm.shape[-3:]
            heatmaps.append(hm)

        heatmaps = torch.cat(heatmaps, dim=0) #B*T, N, H, W
        return heatmaps.reshape(B,T,N,H,W) 


    def apply_spatial_ops(self, x, spatial_ops_func):
        ndim = x.ndim
        if ndim > 4:
            B, T, C_, H, W = x.shape
            x = x.view(-1, C_, H, W)
        chunks = torch.split(x, self.preprocess_chunksize, dim=0)
        transformed_x = []
        for chunk in chunks:
            transformed_x.append(spatial_ops_func(chunk))
        _, C_, H_o, W_o = transformed_x[-1].shape
        transformed_x = torch.cat(transformed_x, dim=0)
        if ndim > 4:
            transformed_x = transformed_x.view(B, T, C_, H_o, W_o)
        return transformed_x    


    def augment_preprocess_inputs(self, is_train, sgn_videos=None, sgn_heatmaps=None, sgn_videos_low=None, sgn_heatmaps_low=None):
        rgb_h, rgb_w = self.transform_cfg.get('img_size',224), self.transform_cfg.get('img_size',224)
        if sgn_heatmaps!=None:
            hm_h, hm_w = self.heatmap_cfg['input_size'], self.heatmap_cfg['input_size']
            #factor_h, factor_w= hm_h/rgb_h, hm_w/rgb_w ！！
            if sgn_videos!=None:
                hm_h0, hm_w0 = sgn_heatmaps.shape[-2],sgn_heatmaps.shape[-1]  #B,T,C,H,W
                B, T, C, rgb_h0, rgb_w0 = sgn_videos.shape  #B,T,C,H,W
                factor_h, factor_w= hm_h0/rgb_h0, hm_w0/rgb_w0 # ！！
        if is_train:
            if sgn_videos!=None:
                if self.transform_cfg.get('color_jitter',False) and random.random()<0.3:
                    color_jitter_op = torchvision.transforms.ColorJitter(0.4,0.4,0.4,0.1)
                    sgn_videos = color_jitter_op(sgn_videos) # B T C H W
                    if self.input_streams == ['rgb', 'rgb']:
                        sgn_heatmaps = color_jitter_op(sgn_heatmaps)
                    if sgn_videos_low is not None:
                        sgn_videos_low = color_jitter_op(sgn_videos_low)
                if self.transform_cfg.get('gaussian_blur',False):
                    gaussian_blur_op = torchvision.transforms.GaussianBlur(kernel_size=3)
                    sgn_videos = gaussian_blur_op(sgn_videos.view(-1, C, rgb_h0, rgb_w0)) # B T C H W
                    sgn_videos = sgn_videos.view(B,T,C,rgb_h0,rgb_w0)
                if self.transform_cfg.get('random_aug', False):
                    randaug_op = torchvision.transforms.RandAugment()
                    sgn_videos = randaug_op(sgn_videos.view(-1, C, rgb_h0, rgb_w0)) # B T C H W
                    sgn_videos = sgn_videos.view(B,T,C,rgb_h0,rgb_w0)
                elif self.transform_cfg.get('random_resized_crop', True):
                    i,j,h,w = torchvision.transforms.RandomResizedCrop.get_params(
                        img=sgn_videos,
                        scale=(self.transform_cfg.get('bottom_area',0.2), 1.0), 
                        ratio=(self.transform_cfg.get('aspect_ratio_min',3./4), 
                            self.transform_cfg.get('aspect_ratio_max',4./3)))
                    sgn_videos = self.apply_spatial_ops(
                        sgn_videos, 
                        spatial_ops_func=lambda x:torchvision.transforms.functional.resized_crop(
                            x, i, j, h, w, [rgb_h, rgb_w]))
                    if sgn_videos_low is not None:
                        sgn_videos_low = self.apply_spatial_ops(
                            sgn_videos_low, 
                            spatial_ops_func=lambda x:torchvision.transforms.functional.resized_crop(
                                x, i, j, h, w, [rgb_h, rgb_w]))
                else:
                    i,j,h,w = torchvision.transforms.RandomCrop.get_params(img=sgn_videos, output_size=[rgb_h, rgb_w])
                    sgn_videos = self.apply_spatial_ops(
                        sgn_videos, 
                        spatial_ops_func=lambda x:torchvision.transforms.functional.crop(
                            x, i, j, h, w))

            if sgn_heatmaps!=None:
                if sgn_videos!=None and not self.transform_cfg.get('random_aug', False):
                    i2, j2, h2, w2 = int(i*factor_h), int(j*factor_w), int(h*factor_h), int(w*factor_w)
                else:
                    i2, j2, h2, w2 = torchvision.transforms.RandomResizedCrop.get_params(
                        img=sgn_heatmaps,
                        scale=(self.transform_cfg.get('bottom_area',0.2), 1.0), 
                        ratio=(self.transform_cfg.get('aspect_ratio_min',3./4), 
                            self.transform_cfg.get('aspect_ratio_max',4./3)))
                if self.transform_cfg.get('random_resized_crop', True):
                    sgn_heatmaps = self.apply_spatial_ops(
                            sgn_heatmaps,
                            spatial_ops_func=lambda x:torchvision.transforms.functional.resized_crop(
                            x, i2, j2, h2, w2, [hm_h, hm_w]))
                    if sgn_heatmaps_low is not None:
                        sgn_heatmaps_low = self.apply_spatial_ops(
                            sgn_heatmaps_low,
                            spatial_ops_func=lambda x:torchvision.transforms.functional.resized_crop(
                            x, i2, j2, h2, w2, [hm_h, hm_w]))
                else:
                    sgn_heatmaps = self.apply_spatial_ops(
                                    sgn_heatmaps, 
                                    spatial_ops_func=lambda x:torchvision.transforms.functional.crop(
                                        x, i2, j2, h2, w2))
                    # need to resize to 112x112
                    sgn_heatmaps = self.apply_spatial_ops(
                                    sgn_heatmaps, 
                                    spatial_ops_func=lambda x:torchvision.transforms.functional.resize(x, [hm_h, hm_w]))

        else:
            if sgn_videos != None:
                spatial_ops = []
                if self.transform_cfg.get('center_crop', False)==True:
                    spatial_ops.append(torchvision.transforms.CenterCrop(
                        self.transform_cfg['center_crop_size']))
                spatial_ops.append(torchvision.transforms.Resize([rgb_h, rgb_w]))
                spatial_ops = torchvision.transforms.Compose(spatial_ops)
                sgn_videos = self.apply_spatial_ops(sgn_videos, spatial_ops)
                if sgn_videos_low is not None:
                    sgn_videos_low = self.apply_spatial_ops(sgn_videos_low, spatial_ops)
            if sgn_heatmaps != None:
                spatial_ops = []
                if self.transform_cfg.get('center_crop', False)==True:
                    spatial_ops.append(
                        torchvision.transforms.CenterCrop(
                            [int(self.transform_cfg['center_crop_size']*factor_h),
                            int(self.transform_cfg['center_crop_size']*factor_w)]))
                spatial_ops.append(torchvision.transforms.Resize([hm_h, hm_w]))
                spatial_ops = torchvision.transforms.Compose(spatial_ops)
                sgn_heatmaps = self.apply_spatial_ops(sgn_heatmaps, spatial_ops)
                if sgn_heatmaps_low is not None:
                    sgn_heatmaps_low = self.apply_spatial_ops(sgn_heatmaps_low, spatial_ops)

        if sgn_videos!=None:
            #convert to BGR for S3D
            if 'r3d' not in self.cfg and 'x3d' not in self.cfg and 'vit' not in self.cfg:
                sgn_videos = sgn_videos[:,:,[2,1,0],:,:] # B T 3 H W
            sgn_videos = sgn_videos.float()
            sgn_videos = (sgn_videos-0.5)/0.5
            sgn_videos = sgn_videos.permute(0,2,1,3,4).float() # B C T H W
        if sgn_videos_low!=None:
            #convert to BGR for S3D
            if 'r3d' not in self.cfg and 'x3d' not in self.cfg and 'vit' not in self.cfg:
                sgn_videos_low = sgn_videos_low[:,:,[2,1,0],:,:] # B T 3 H W
            sgn_videos_low = sgn_videos_low.float()
            sgn_videos_low = (sgn_videos_low-0.5)/0.5
            sgn_videos_low = sgn_videos_low.permute(0,2,1,3,4).float() # B C T H W
        if sgn_heatmaps!=None:
            sgn_heatmaps = (sgn_heatmaps-0.5)/0.5
            if sgn_heatmaps.ndim > 4:
                sgn_heatmaps = sgn_heatmaps.permute(0,2,1,3,4).float()
            else:
                sgn_heatmaps = sgn_heatmaps.float()
        if sgn_heatmaps_low!=None:
            sgn_heatmaps_low = (sgn_heatmaps_low-0.5)/0.5
            if sgn_heatmaps_low.ndim > 4:
                sgn_heatmaps_low = sgn_heatmaps_low.permute(0,2,1,3,4).float()
            else:
                sgn_heatmaps_low = sgn_heatmaps_low.float()
        return sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low
    

    def mixup(self, mixup_param, ip_a, ip_b, labels, do_joint_mixup=False, ip_c=None, ip_d=None):
        #initialize
        mix_a, mix_b, mix_c, mix_d = ip_a, ip_b, ip_c, ip_d
        y_a = y_b = labels
        lam = 0
        index = torch.arange(labels.shape[0])
        do_mixup = False
        
        if mixup_param and self.training:
            prob, alpha = map(float, mixup_param.split('_'))
            if random.random() < prob or do_joint_mixup:
                # do mixup
                do_mixup = True
                lam = np.random.beta(alpha, alpha)
                batch_size = ip_a.shape[0] if ip_a is not None else ip_b.shape[0]
                index = torch.randperm(batch_size)
                index = index.to(ip_a.device) if ip_a is not None else index.to(ip_b.device)

                if ip_a is not None:
                    mix_a = lam * ip_a + (1. - lam) * ip_a[index]
                if ip_b is not None:
                    mix_b = lam * ip_b + (1. - lam) * ip_b[index]
                if ip_c is not None:
                    mix_c = lam * ip_c + (1. - lam) * ip_c[index]
                if ip_d is not None:
                    mix_d = lam * ip_d + (1. - lam) * ip_d[index]
                y_a, y_b = labels, labels[index]

        return mix_a, mix_b, mix_c, mix_d, y_a, y_b, lam, index, do_mixup


    def _forward_impl(self, is_train, labels, sgn_videos=None, sgn_keypoints=None, epoch=0, **kwargs):
        s3d_outputs = []
        # Preprocess (Move from data loader)
        with torch.no_grad():
            #1. generate heatmaps
            if 'keypoint' in self.input_streams:
                assert sgn_keypoints != None
                sgn_heatmaps = self.generate_batch_heatmap(sgn_keypoints, self.heatmap_cfg) #B,T,N,H,W or B,N,H,W
            else:
                sgn_heatmaps = None
            
            if not 'rgb' in self.input_streams:
                sgn_videos = None
            
            if self.input_streams == ['rgb', 'rgb']:
                sgn_heatmaps = sgn_keypoints
            
            sgn_videos_low = kwargs.pop('sgn_videos_low', None)
            sgn_keypoints_low = kwargs.pop('sgn_keypoints_low', None)
            sgn_heatmaps_low = None
            if len(self.input_streams)==4:
                sgn_heatmaps_low = self.generate_batch_heatmap(sgn_keypoints_low, self.heatmap_cfg)
            
            #2. augmentation and permute(colorjitter, randomresizedcrop/centercrop+resize, normalize-0.5~0.5, channelswap for RGB)
            sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low = self.augment_preprocess_inputs(is_train, sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low)

            #mixup
            mixup_param = self.transform_cfg.get('mixup', None)
            sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low, y_a, y_b, lam, index, do_mixup = \
                self.mixup(mixup_param, sgn_videos, sgn_heatmaps, labels, ip_c=sgn_videos_low, ip_d=sgn_heatmaps_low)

        if self.input_streams == ['rgb']:
            s3d_outputs = self.visual_backbone(sgn_videos=sgn_videos)
        elif self.input_streams == ['keypoint']:
            s3d_outputs = self.visual_backbone_keypoint(sgn_videos=sgn_heatmaps)
        elif len(self.input_streams)==2:
            s3d_outputs = self.visual_backbone_twostream(x_rgb=sgn_videos, x_pose=sgn_heatmaps)
        elif len(self.input_streams)==4:
            s3d_outputs = self.visual_backbone_fourstream(sgn_videos, sgn_videos_low, sgn_heatmaps, sgn_heatmaps_low)

        if self.fuse_method is None:
            assert len(self.input_streams)==1, self.input_streams
            assert self.cfg['pyramid']['rgb'] == self.cfg['pyramid']['pose']
            if 'rgb' in self.input_streams:
                head_outputs = self.visual_head(x=s3d_outputs['sgn_feature'], labels=labels)
            
            elif 'keypoint' in self.input_streams:
                head_outputs = self.visual_head_keypoint(x=s3d_outputs['sgn_feature'], labels=labels)

            else:
                raise ValueError
            outputs = {**head_outputs}

        elif 'sephead' in self.fuse_method or 'triplehead' in self.fuse_method:
            assert len(self.input_streams)==2
            # rgb
            head_outputs_rgb = self.visual_head(x=s3d_outputs['rgb_fea_lst'][-1], labels=labels)
            head_rgb_input = s3d_outputs['rgb_fea_lst'][-1]  #B,C,T,H,W

            # keypoint
            head_keypoint_input = s3d_outputs['pose_fea_lst'][-1]
            head_outputs_keypoint = self.visual_head_keypoint(x=s3d_outputs['pose_fea_lst'][-1], labels=labels)

            head_outputs = {'gloss_logits': None, 
                            'rgb_gloss_logits': head_outputs_rgb['gloss_logits'],
                            'keypoint_gloss_logits': head_outputs_keypoint['gloss_logits'],
                            'gloss_probabilities': None,
                            'rgb_gloss_probabilities': head_outputs_rgb['gloss_probabilities'],
                            'keypoint_gloss_probabilities': head_outputs_keypoint['gloss_probabilities'],
                            'head_rgb_input': head_rgb_input, 
                            'head_keypoint_input': head_keypoint_input,
                            }
            
            if 'triplehead' in self.fuse_method:
                assert self.visual_head_fuse != None
                if self.input_streams == ['rgb', 'rgb']:
                    fused_sgn_features = torch.cat([head_rgb_input.mean(dim=1), head_keypoint_input.mean(dim=1)], dim=-1)
                else:
                    fused_sgn_features = torch.cat([head_rgb_input, head_keypoint_input], dim=-1)

                head_outputs_fuse = self.visual_head_fuse(x=fused_sgn_features, labels=labels) 
                head_outputs['fuse_gloss_probabilities'] = head_outputs_fuse['gloss_probabilities']
                head_outputs['fuse_gloss_logits'] = head_outputs_fuse['gloss_logits']
                head_outputs['fuse_gloss_feature'] = head_outputs_fuse['gloss_feature']
                head_outputs['head_fuse_input'] = fused_sgn_features

                head_outputs['rgb_topk_idx'] = head_outputs_rgb['topk_idx']
                head_outputs['keypoint_topk_idx'] = head_outputs_keypoint['topk_idx']
                head_outputs['fuse_topk_idx'] = head_outputs_fuse['topk_idx']

                head_outputs['ensemble_last_gloss_logits'] = (head_outputs['fuse_gloss_logits'].softmax(dim=-1)+\
                    head_outputs['rgb_gloss_logits'].softmax(dim=-1)+head_outputs['keypoint_gloss_logits'].softmax(-1)).log()
                head_outputs['ensemble_last_gloss_raw_logits'] = head_outputs['ensemble_last_gloss_logits']
            
            else:
                raise ValueError 
            
            head_outputs['ensemble_last_gloss_probabilities_log'] = head_outputs['ensemble_last_gloss_logits'].log_softmax(1) 
            head_outputs['ensemble_last_gloss_probabilities'] = head_outputs['ensemble_last_gloss_logits'].softmax(1)
            outputs = {**head_outputs}
        
        elif 'four' in self.fuse_method:
            outputs = {}
            keys_of_int = ['gloss_logits', 'word_fused_gloss_logits', 'topk_idx']
            for head_name, fea in s3d_outputs.items():
                head_ops = self.head_dict[head_name](x=fea, labels=labels)
                for k in keys_of_int:
                    outputs[head_name+'_'+k] = head_ops[k]

            # deal with fuse heads
            effect_head_lst = ['rgb-h', 'rgb-l', 'kp-h', 'kp-l']
            for head_name, head in self.head_dict.items():
                if head is None or 'fuse' not in head_name:
                    continue
                effect_head_lst.append(head_name)
                if head_name == 'fuse':
                    fused_fea = torch.cat([s3d_outputs['rgb-h'], s3d_outputs['rgb-l'], s3d_outputs['kp-h'], s3d_outputs['kp-l']], dim=-1)
                    head_ops = head(x=fused_fea, labels=labels)
                elif head_name == 'fuse-h':
                    fused_fea = torch.cat([s3d_outputs['rgb-h'], s3d_outputs['kp-h']], dim=-1)
                    head_ops = head(x=fused_fea, labels=labels)
                elif head_name == 'fuse-l':
                    fused_fea = torch.cat([s3d_outputs['rgb-l'], s3d_outputs['kp-l']], dim=-1)
                    head_ops = head(x=fused_fea, labels=labels)
                elif head_name == 'fuse-x-rgb':
                    fused_fea = torch.cat([s3d_outputs['rgb-h'], s3d_outputs['rgb-l']], dim=-1)
                    head_ops = head(x=fused_fea, labels=labels)
                elif head_name == 'fuse-x-kp':
                    fused_fea = torch.cat([s3d_outputs['kp-h'], s3d_outputs['kp-l']], dim=-1)
                    head_ops = head(x=fused_fea, labels=labels)
                for k in keys_of_int:
                    outputs[head_name+'_'+k] = head_ops[k]
            del head_ops
            
            # ensemble prob and logits
            for head_name, head in self.head_dict.items():
                if head is None:
                    continue
                outputs['ensemble_all_gloss_logits'] = outputs.get('ensemble_all_gloss_logits', torch.zeros_like(outputs['rgb-h_gloss_logits']))\
                                                        + outputs[head_name+'_gloss_logits'].softmax(dim=-1)
                if 'fuse' in head_name:
                    outputs['ensemble_last_gloss_logits'] = outputs.get('ensemble_last_gloss_logits', torch.zeros_like(outputs['rgb-h_gloss_logits']))\
                                                            + outputs[head_name+'_gloss_logits'].softmax(dim=-1)
            outputs['ensemble_all_gloss_logits'] = outputs['ensemble_all_gloss_logits'].log()
            outputs['ensemble_last_gloss_logits'] = outputs['ensemble_last_gloss_logits'].log()

        else:
            raise ValueError
        
        # compute losses
        if self.fuse_method == None or 'loss1' in self.fuse_method:
            if head_outputs['gloss_logits'] is not None:
                if mixup_param:
                    outputs['recognition_loss'] = lam*self.compute_recognition_loss(logits=head_outputs['gloss_logits'], labels=y_a) + \
                                                (1.-lam)*self.compute_recognition_loss(logits=head_outputs['gloss_logits'], labels=y_b)
                else:
                    outputs['recognition_loss'] = self.compute_recognition_loss(logits=head_outputs['gloss_logits'], labels=labels)
        
        elif 'loss2' in self.fuse_method or 'loss3' in self.fuse_method or 'triplehead' in self.fuse_method or 'four' in self.fuse_method:
            assert len(self.input_streams)==2 or len(self.input_streams)==4
            if len(self.input_streams)==2:
                key_lst = ['rgb', 'keypoint', 'fuse']
            else:
                key_lst = effect_head_lst
                head_outputs = outputs
            for k in key_lst:
                if f'{k}_gloss_logits' in head_outputs:
                    if mixup_param:
                        outputs[f'recognition_loss_{k}'] = lam*self.compute_recognition_loss(
                                                                    logits=head_outputs[f'{k}_gloss_logits'], 
                                                                    labels=y_a, 
                                                                    head_name=k
                                                                    ) + \
                                                            (1.-lam)*self.compute_recognition_loss(
                                                                    logits=head_outputs[f'{k}_gloss_logits'], 
                                                                    labels=y_b, 
                                                                    head_name=k
                                                                    )
                    else:
                        outputs[f'recognition_loss_{k}'] = self.compute_recognition_loss(
                            logits=head_outputs[f'{k}_gloss_logits'],
                            labels=labels)

            if 'loss2' in self.fuse_method:
                outputs['recognition_loss'] = outputs['recognition_loss_rgb'] + outputs['recognition_loss_keypoint']

            elif 'triplehead' in self.fuse_method:
                outputs['recognition_loss'] = outputs['recognition_loss_rgb'] + outputs['recognition_loss_keypoint'] + outputs['recognition_loss_fuse']
            
            elif 'four' in self.fuse_method:
                for k in effect_head_lst:
                    outputs['recognition_loss'] = outputs.get('recognition_loss', torch.tensor(0.0).to(sgn_videos.device))\
                                                + outputs[f'recognition_loss_{k}']

        else:
            raise ValueError
        outputs['total_loss'] = outputs['recognition_loss']

        if self.contras_setting is not None:
            if len(self.input_streams) == 2:
                key_lst = ['rgb_', 'keypoint_', 'fuse_']
            elif len(self.input_streams) == 4:
                key_lst = [e+'_' for e in effect_head_lst]
            else:
                key_lst = ['']
            for k in key_lst:
                contras_loss = torch.tensor(0.0).to(sgn_videos.device) if sgn_videos is not None else torch.tensor(0.0).to(sgn_keypoints.device)
                if 'dual' in self.contras_setting:
                    word_fused_gloss_logits = outputs[f'{k}word_fused_gloss_logits']
                    if word_fused_gloss_logits is None:
                        # only apply to some heads
                        continue
                    topk_idx = outputs[f'{k}topk_idx']
                    if mixup_param:
                        contras_loss = self.contras_loss_func(word_fused_gloss_logits, labels, topk_idx=topk_idx, mixup_lam=lam, y_a=y_a, y_b=y_b)
                    else:
                        contras_loss = self.contras_loss_func(word_fused_gloss_logits, labels, topk_idx=topk_idx)
                
                outputs[f'contras_loss_{k}'] = contras_loss

                #contras only "=" other wise "+="
                if 'cosine' in self.contras_setting:
                    cur_loss_weight = 0.5 * self.contras_loss_weight * (1.0 + math.cos(epoch*math.pi/100))
                    outputs['total_loss'] = outputs['total_loss'] + cur_loss_weight * contras_loss
                else:
                    outputs['total_loss'] = outputs['total_loss'] + self.contras_loss_weight * contras_loss

        return outputs


    def forward(self, is_train, labels, sgn_videos=None, sgn_keypoints=None, epoch=0, **kwargs):
        if len(sgn_videos) == 1:
            return self._forward_impl(is_train, labels, sgn_videos=sgn_videos[0], sgn_keypoints=sgn_keypoints[0], epoch=epoch, **kwargs)

        else:
            if len(self.input_streams) == 4:
                return self._forward_impl(is_train, labels, sgn_videos=sgn_videos[0], sgn_keypoints=sgn_keypoints[0], epoch=epoch,
                                        sgn_videos_low=sgn_videos[1], sgn_keypoints_low=sgn_keypoints[1], **kwargs)
            
            elif len(self.input_streams) == 2:
                sgn_videos, sgn_keypoints = sgn_videos
                return self._forward_impl(is_train, labels, sgn_videos=sgn_videos, sgn_keypoints=sgn_keypoints, epoch=epoch, **kwargs)