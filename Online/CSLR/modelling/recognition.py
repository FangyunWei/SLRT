import torch
from modelling.S3D import S3D_backbone
from modelling.trajectory import TrajModel
from modelling.resnet2d import resnet50, ResNet
from modelling.R3D import get_resnet3d
from modelling.X3D import get_x3d
# from modelling.resnet3d import ResNet3dSlowOnly_backbone
from modelling.two_stream import S3D_two_stream_v2
from modelling.four_stream import S3D_four_stream
from modelling.vit import build_vit
from utils.misc import get_logger, neq_load_customized, upd_MAE_ckpt_keys
import glob, os, random, torchvision
from itertools import groupby
from modelling.Visualhead import VisualHead, SepConvVisualHead, MarginVisualHead, WeightLearner
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from utils.gen_gaussian import gen_gaussian_hmap_op
import math
from utils.loss import LabelSmoothCE, BCEwithWordSim


class RecognitionNetwork(torch.nn.Module):
    def __init__(self, cfg, transform_cfg, cls_num=2000, input_streams=['rgb'], input_frames=64, word_emb_tab=None):
        super().__init__()
        logger = get_logger()
        self.cfg = cfg
        self.bag_size = transform_cfg.get('bag_size', 6)
        self.num_instance = transform_cfg.get('num_instance', 6)
        self.input_streams = input_streams
        self.fuse_method = cfg.get('fuse_method', None)
        self.heatmap_cfg = cfg.get('heatmap_cfg', {})
        self.traj_hmap_cfg = cfg.get('traj_hmap_cfg', {})
        self.transform_cfg = transform_cfg
        self.preprocess_chunksize = self.heatmap_cfg.get('preprocess_chunksize', 16)
        self.word_emb_tab = word_emb_tab
        if self.heatmap_cfg.get('center', None) is not None and self.heatmap_cfg.get('rela_comb', False):
            cfg['keypoint_s3d']['in_channel'] = cfg['keypoint_s3d']['in_channel'] * (1+len(self.heatmap_cfg['center']))
        
        cfg['pyramid'] = cfg.get('pyramid', {'version':None, 'rgb':None, 'pose':None})
        self.visual_backbone = self.visual_backbone_keypoint = self.visual_backbone_twostream = None
        if input_streams == ['rgb']:
            if 'vit' in cfg:
                self.visual_backbone = build_vit(num_cls=cls_num, num_frames=input_frames, cfg=cfg['vit'], in_channels=3, img_size=224)
            elif 'r3d' in cfg:
                self.visual_backbone = get_resnet3d(**cfg['r3d'])
            elif 'x3d' in cfg:
                self.visual_backbone = get_x3d(**cfg['x3d'])
            elif 's3d' in cfg:
                self.visual_backbone = S3D_backbone(in_channel=3, **cfg['s3d'], cfg_pyramid=cfg['pyramid'])
            else:
                raise ValueError
            
        elif input_streams == ['keypoint']:
            if 'vit' in cfg:
                self.visual_backbone_keypoint = build_vit(num_cls=cls_num, num_frames=input_frames, cfg=cfg['vit'], 
                        in_channels=cfg['keypoint_s3d']['in_channel'], img_size=cfg['heatmap_cfg']['input_size'])
            elif 'keypoint_s3d' in cfg:
                self.visual_backbone_keypoint = S3D_backbone(**cfg['keypoint_s3d'], cfg_pyramid=cfg['pyramid'])
            else:
                raise ValueError
            
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
                fusion_features=cfg['lateral'].get('fusion_features',['c1','c2','c3']),
                word_emb_tab=None,
                word_emb_dim=cfg['visual_head']['word_emb_dim'],
                temp=cfg['visual_head']['temp'],
                contras_setting=cfg['visual_head']['contras_setting'],
                coord_conv=cfg['keypoint_s3d'].get('coord_conv', None)
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
                fusion_features=cfg['lateral'].get('fusion_features',['c1','c2','c3']),
                word_emb_tab=None,
                word_emb_dim=cfg['visual_head']['word_emb_dim'],
                temp=cfg['visual_head']['temp'],
                contras_setting=cfg['visual_head']['contras_setting'],
                coord_conv=cfg['keypoint_s3d'].get('coord_conv', None)
            )
            
        if 'trajectory' in input_streams:
            traj_cfg = cfg.get('traj_cfg', {})
            part2index = {'hand': list(range(42)), 'mouth_half': list(range(42,52)), 'pose': list(range(52,63))}
            parts = traj_cfg.pop('parts', ['hand'])
            self.kp_idx = []
            for p in sorted(parts):
                self.kp_idx.extend(part2index[p])
            self.cen_idx = [52, 53, 54, 55, 56, 57, 58]
            if self.traj_hmap_cfg == {}:
                traj_type = traj_cfg.get('type', '2d')
                if '1d' in traj_type: 
                    init_planes = len(self.kp_idx) * 2
                    if 'rela' in traj_type:
                        init_planes *= len(self.cen_idx)
                    self.traj_backbone = TrajModel(init_planes=init_planes, heatmap_cfg=self.heatmap_cfg, **traj_cfg)
                elif '2d' in traj_type:
                    init_planes = 3 if traj_type == '2d' else len(self.cen_idx) * 2
                    self.traj_backbone = resnet50(pretrained_path='../../pretrained_models/resnet50-11ad3fa6.pth', 
                                            init_planes=init_planes)
            else:
                self.traj_backbone = resnet50(pretrained_path='../../pretrained_models/resnet50-11ad3fa6.pth', 
                                            init_planes=63)
            self.traj_cfg = traj_cfg

        if 'visual_head' in cfg:
            head_var = cfg['visual_head'].get('variant', 'vanilla')
            if head_var == 'vanilla':
                HeadCLS = VisualHead
            elif head_var == 'sep_conv':
                HeadCLS = SepConvVisualHead
            elif head_var in ['arcface', 'cosface']:
                HeadCLS = MarginVisualHead
            language_apply_to = cfg.get('language_apply_to', 'rgb_keypoint_joint')
            if 'rgb' in input_streams or len(self.input_streams)==2:
                rgb_head_cfg = deepcopy(cfg['visual_head'])
                if 'rgb' not in language_apply_to:
                    rgb_head_cfg['contras_setting'] = None
                self.visual_head = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **rgb_head_cfg)
                
                # num_remain_heads = cfg['aux_head']['num']
                # if num_remain_heads>0:
                #     dims = [192, 480, 832]
                #     dims = dims[-num_remain_heads:]
                #     self.visual_head_remain = torch.nn.ModuleList()
                #     for i in range(num_remain_heads):
                #         new_cfg = deepcopy(cfg['visual_head'])
                #         new_cfg['input_size'] = dims[i]
                #         self.visual_head_remain.append(HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **new_cfg))
            else:
                self.visual_head = None
            
            if 'keypoint' in input_streams or 'keypoint_coord' in input_streams or len(self.input_streams)==2:
                keypoint_head_cfg = deepcopy(cfg['visual_head'])
                if 'keypoint' not in language_apply_to:
                    keypoint_head_cfg['contras_setting'] = None
                self.visual_head_keypoint = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **keypoint_head_cfg)
                
                # num_remain_heads = cfg['aux_head']['num']
                # if num_remain_heads>0:
                #     dims = [192, 480, 832]
                #     dims = dims[-num_remain_heads:]
                #     self.visual_head_keypoint_remain = torch.nn.ModuleList()
                #     for i in range(num_remain_heads):
                #         new_cfg = deepcopy(cfg['visual_head'])
                #         new_cfg['input_size'] = dims[i]
                #         self.visual_head_keypoint_remain.append(HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **new_cfg))
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
                if 'trajectory' in self.input_streams:
                    if isinstance(self.traj_backbone, ResNet):
                        joint_head_cfg['input_size'] = joint_head_cfg['input_size'] + 2048
                    else:
                        joint_head_cfg['input_size'] = joint_head_cfg['input_size'] + traj_cfg['dim']
                self.visual_head_fuse = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **joint_head_cfg)
                if cfg['visual_head'].get('weighted', False):
                    self.head_prob_weight_learner = WeightLearner(cfg['visual_head']['input_size'], num_inputs=3)
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
            
            if 'trajectory' in input_streams:
                traj_head_cfg = deepcopy(cfg['visual_head'])
                if isinstance(self.traj_backbone, ResNet):
                    traj_head_cfg['input_size'] = 2048
                    traj_head_cfg['cnn_type'] = '2d'
                else:
                    traj_head_cfg['input_size'] = traj_cfg['dim']
                    traj_head_cfg['cnn_type'] = '1d'
                if 'traj' not in language_apply_to:
                    traj_head_cfg['contras_setting'] = None
                self.visual_head_traj = HeadCLS(cls_num=cls_num, word_emb_tab=word_emb_tab, **traj_head_cfg)

        if 'pretrained_path_rgb' in cfg:
            load_dict = torch.load(cfg['pretrained_path_rgb'],map_location='cpu')['model_state']      
            backbone_dict, head_dict, head_remain_dict = {}, {}, {}
            for k, v in load_dict.items():
                if 'visual_backbone' in k:
                    backbone_dict[k.replace('recognition_network.visual_backbone.','')] = v
                if 'visual_head' in k and 'visual_head_remain' not in k:
                    head_dict[k.replace('recognition_network.visual_head.','')] = v
                if 'visual_head_remain' in k:
                    head_remain_dict[k.replace('recognition_network.visual_head_remain.','')] = v
            if self.visual_backbone!=None and self.visual_backbone_twostream==None:
                neq_load_customized(self.visual_backbone, backbone_dict, verbose=True)
                neq_load_customized(self.visual_head, head_dict, verbose=True)
                logger.info('Load visual_backbone and visual_head for rgb from {}'.format(cfg['pretrained_path_rgb']))
            elif self.visual_backbone==None and self.visual_backbone_twostream!=None:
                neq_load_customized(self.visual_backbone_twostream.rgb_stream, backbone_dict, verbose=True)
                neq_load_customized(self.visual_head, head_dict, verbose=True)
                if cfg['pyramid']['rgb'] == 'multi_head':
                    neq_load_customized(self.visual_head_remain, head_remain_dict, verbose=True)
                logger.info('Load visual_backbone_twostream.rgb_stream and visual_head for rgb from {}'.format(cfg['pretrained_path_rgb'])) 
            else:
                logger.info('No rgb stream exists in the network')

        if 'pretrained_path_keypoint' in cfg and input_streams != ['keypoint_coord']:
            load_dict = torch.load(cfg['pretrained_path_keypoint'], map_location='cpu')['model_state']
            if 'mae' in cfg['pretrained_path_keypoint']:
                load_dict = upd_MAE_ckpt_keys(load_dict)
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
                if cfg['pyramid']['pose'] == 'multi_head':
                    neq_load_customized(self.visual_head_keypoint_remain, head_remain_dict, verbose=True)
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
        
        if 'pretrained_path_trajectory' in cfg and 'trajectory' in self.input_streams:
            load_dict = torch.load(cfg['pretrained_path_trajectory'],map_location='cpu')['model_state']
            backbone_dict, head_dict = {}, {}
            for k, v in load_dict.items():
                if 'traj_backbone' in k:
                    backbone_dict[k.replace('recognition_network.traj_backbone.','')] = v
                if 'visual_head_traj' in k: #for model trained using new_code
                    head_dict[k.replace('recognition_network.visual_head_traj.','')] = v
            neq_load_customized(self.traj_backbone, backbone_dict, verbose=True)
            neq_load_customized(self.visual_head_traj, head_dict, verbose=True)
            logger.info('Load traj_backbone and visual_head for traj from {}'.format(cfg['pretrained_path_trajectory']))

        label_smooth = cfg.get('label_smooth', 0.0)
        blank_weight = cfg.get('blank_weight', None)
        aug_weight = cfg.get('aug_weight', None)  #loss weight for augmented data
        self.bag_loss_flags = cfg.get('bag_loss', [False, False])  #for main cls loss and split loss
        if type(label_smooth)==float and label_smooth > 0:
            if blank_weight is None:
                cls_weight = None
            else:
                cls_weight = [blank_weight] + [1.0]*(cls_num-1)
                cls_weight = torch.tensor(cls_weight)
            self.recognition_loss_func = LabelSmoothCE(lb_smooth=label_smooth, reduction='mean', 
                                                        cls_weight=cls_weight, aug_weight=aug_weight, 
                                                        bag_size=self.bag_size, num_instance=self.num_instance)
        elif type(label_smooth)==str:
            temp, lb_smooth, norm_type = float(label_smooth.split('_')[-1]), float(label_smooth.split('_')[-2]), label_smooth.split('_')[-3]
            if 'word_emb_sim' in label_smooth:
                variant = 'word_sim'
                if 'vis' in label_smooth:
                    variant = 'word_sim_vis'
                elif 'xmodal' in label_smooth:
                    variant = label_smooth.replace('_emb', '')
                self.recognition_loss_func = LabelSmoothCE(lb_smooth=lb_smooth, reduction='mean', 
                            word_emb_tab=word_emb_tab, norm_type=norm_type, temp=temp, variant=variant)
            elif 'iou_soft' in label_smooth:
                #iou_soft_l1_0.2_1.0
                self.recognition_loss_func = LabelSmoothCE(lb_smooth=lb_smooth, reduction='mean', norm_type=norm_type, temp=temp, variant='iou_soft')
        else:
            self.recognition_loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.contras_loss_weight = cfg.get('contras_loss_weight', 0.0)
        if type(self.contras_loss_weight)==float and self.contras_loss_weight > 0:
            self.contras_loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
        self.contras_setting = cfg['visual_head'].get('contras_setting', None)
        if self.contras_setting:
            if 'dual' in self.contras_setting:
                if 'multi_label' in self.contras_setting:
                    self.contras_loss_func = BCEwithWordSim('mean', word_emb_tab=word_emb_tab)
                elif 'word_sim' not in self.contras_setting:
                    if 'margin' in self.contras_setting:
                        self.contras_loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
                    else:
                        self.contras_loss_func = LabelSmoothCE(reduction='mean', variant=self.contras_setting)
                # else:
                #     self.contras_loss_func = LabelSmoothCE(reduction='mean', variant='dual_word_sim', word_emb_tab=word_emb_tab)
            if 'contras' in self.contras_setting:
                self.contras_loss_func2 = torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.head_split_setting = cfg['visual_head'].get('split_setting', None)
        if self.head_split_setting is not None and ('cam' in self.head_split_setting or 'att' in self.head_split_setting):
            self.kl_loss_func = torch.nn.KLDivLoss(reduction='batchmean')


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
        if heatmap_cfg.get('temp_merge', False):
            self.preprocess_chunksize = B
        chunk_size = int(math.ceil((B*T)/self.preprocess_chunksize))
        chunks = torch.split(keypoints, chunk_size, dim=0)

        # raw_h, raw_w = heatmap_cfg['raw_size'] #260,210
        # if 'map_size' not in heatmap_cfg:
        #     map_h, map_w = raw_h, raw_w
        # else:
        #     map_h, map_w = heatmap_cfg['map_size']

        # if heatmap_cfg.get('temp_merge', False):
        #     heatmaps = torch.zeros(B,N,map_h,map_w).to(keypoints.device)
        # else:
        #     if heatmap_cfg.get('center', None) is not None and heatmap_cfg.get('rela_comb', False):
        #         N *= len(heatmap_cfg['center'])+1
        #     heatmaps = torch.zeros(B*T,N,map_h,map_w).to(keypoints.device)

        heatmaps = []
        i = 0
        for chunk in chunks:
            # print(chunk.shape)
            hm = gen_gaussian_hmap_op(
                coords=chunk,  
                **heatmap_cfg) #sigma, confidence, threshold) #B*T,N,H,W
            N, H, W = hm.shape[-3:]
            heatmaps.append(hm)
            # if heatmap_cfg.get('temp_merge', False):
            #     heatmaps[i] = hm
            # else:
            #     heatmaps[i*chunk_size:(i+1)*chunk_size] = hm
            # i += 1
        
        # if not heatmap_cfg.get('temp_merge', False):
        #     heatmaps = heatmaps.reshape(B,T,N,map_h,map_w)
        # return heatmaps

        if heatmap_cfg.get('temp_merge', False):
            heatmaps = torch.stack(heatmaps, dim=0)  #B,N,H,W
            return heatmaps
        else:
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
            p_hflip = self.transform_cfg.get('p_hflip', 0.0)
            random_hflip = 0
            if p_hflip > 0:
                random_hflip = random.random()
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
                
                # if random_hflip < p_hflip:
                #     # print('flip video')
                #     sgn_videos = torchvision.transforms.functional.hflip(sgn_videos)

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
                
                # if random_hflip < p_hflip:
                #     # print('flip hmap')
                #     sgn_heatmaps = torchvision.transforms.functional.hflip(sgn_heatmaps)
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
    

    def mixup(self, mixup_param, ip_a, ip_b, labels, cross=False, cat=False, do_joint_mixup=False, 
            ip_c=None, ip_d=None, low2high=False, start_idx=None, bag_labels=None):
        #initialize
        mix_a, mix_b, mix_c, mix_d = ip_a, ip_b, ip_c, ip_d
        y_a = y_b = labels
        lam = 0
        index = torch.arange(labels.shape[0])
        do_mixup = False
        if mixup_param and cross:
            if cat:
                mix_a = torch.cat([0.5 * ip_a, 0.5 * ip_b], dim=-1)
            else:
                mix_a = 0.5 * ip_a + 0.5 * ip_b
        
        if mixup_param and self.training:
            prob, alpha = map(float, mixup_param.split('_'))
            if random.random() < prob or do_joint_mixup:
                # do mixup
                do_mixup = True
                lam = np.random.beta(alpha, alpha)
                batch_size = ip_a.shape[0] if ip_a is not None else ip_b.shape[0]
                if bag_labels is None:
                    index = torch.randperm(batch_size)
                else:
                    index = []
                    for i in range(batch_size//self.bag_size):
                        index.append(i*self.bag_size + torch.randperm(self.bag_size))
                    index = torch.cat(index)
                    # print(index)
                index = index.to(ip_a.device) if ip_a is not None else index.to(ip_b.device)
                if not cross:
                    if ip_a is not None:
                        if low2high:
                            mix_a = ip_a
                            mix_a[:, :, start_idx:start_idx+32, :, :] = lam * mix_a[:, :, start_idx:start_idx+32, :, :] + (1.-lam) * ip_c[index]
                        else:
                            mix_a = lam * ip_a + (1. - lam) * ip_a[index]
                    if ip_b is not None:
                        if low2high:
                            mix_b = ip_b
                            mix_b[:, :, start_idx:start_idx+32, :, :] = lam * mix_b[:, :, start_idx:start_idx+32, :, :] + (1.-lam) * ip_d[index]
                        else:
                            mix_b = lam * ip_b + (1. - lam) * ip_b[index]
                    if ip_c is not None:
                        mix_c = lam * ip_c + (1. - lam) * ip_c[index]
                    if ip_d is not None:
                        mix_d = lam * ip_d + (1. - lam) * ip_d[index]
                else:
                    assert ip_a is not None and ip_b is not None
                    if cat:
                        mix_a = torch.cat([lam * ip_a, (1. - lam) * ip_b[index]], dim=-1)
                    else:
                        mix_a = lam * ip_a + (1. - lam) * ip_b[index]
                y_a, y_b = labels, labels[index]

        return mix_a, mix_b, mix_c, mix_d, y_a, y_b, lam, index, do_mixup


    def _forward_impl(self, is_train, labels, sgn_videos=None, sgn_keypoints=None, epoch=0, **kwargs):
        s3d_outputs = []
        traj_inputs = None
        bag_labels = kwargs.pop('bag_labels', None)
        iou_labels = kwargs.pop('iou_labels', None)
        temp_idx = kwargs.pop('temp_idx', None)
        # Preprocess (Move from data loader)
        with torch.no_grad():
            #1. generate heatmaps
            if 'keypoint' in self.input_streams or 'trajectory' in self.input_streams:
                assert sgn_keypoints != None
                sgn_heatmaps = self.generate_batch_heatmap(sgn_keypoints, self.heatmap_cfg) #B,T,N,H,W or B,N,H,W
                if 'trajectory' in self.input_streams:
                    if self.traj_hmap_cfg != {}:
                        traj_heatmaps = self.generate_batch_heatmap(sgn_keypoints, self.traj_hmap_cfg)
                        traj_inputs = traj_heatmaps
                    else:
                        conf = sgn_keypoints[:, :, self.kp_idx, 2]
                        centers = sgn_keypoints[:, :, self.cen_idx, :2]
                        sgn_keypoints = sgn_keypoints[:, :, self.kp_idx, :2]
                        hmap_raw_h, hmap_raw_w = self.heatmap_cfg['raw_size']
                        if 'rela' in self.traj_cfg.get('type', '2d'):
                            B, T = sgn_keypoints.shape[:2]
                            sgn_keypoints = sgn_keypoints.unsqueeze(3) - centers.unsqueeze(2)
                            h_low = 0. - centers[..., 1]  #[B,T,len(self.cen_idx)]
                            h_high = hmap_raw_h - centers[..., 1]
                            w_low = 0. - centers[..., 0]
                            w_high = hmap_raw_w - centers[..., 0]
                            sgn_keypoints[..., 1] = sgn_keypoints[..., 1] - h_low.unsqueeze(2) / h_high.unsqueeze(2) - h_low.unsqueeze(2)
                            sgn_keypoints[..., 0] = sgn_keypoints[..., 0] - w_low.unsqueeze(2) / w_high.unsqueeze(2) - w_low.unsqueeze(2)
                        else:
                            sgn_keypoints[..., 1] = sgn_keypoints[..., 1] / hmap_raw_h
                            sgn_keypoints[..., 0] = sgn_keypoints[..., 0] / hmap_raw_w
                        sgn_keypoints = (sgn_keypoints-0.5) / 0.5
                        sgn_keypoints = torch.clamp(sgn_keypoints, min=-1., max=1.)
                        if isinstance(self.traj_backbone, ResNet):
                            if 'rela' in self.traj_cfg.get('type', '2d'):
                                sgn_keypoints = sgn_keypoints.view(B,T,len(self.kp_idx),-1)
                            else:
                                sgn_keypoints = torch.cat([sgn_keypoints, conf.unsqueeze(-1)], dim=-1)
                            sgn_keypoints = sgn_keypoints.permute(0,3,1,2).contiguous()  #remove inf?
                        else:
                            sgn_keypoints = sgn_keypoints.flatten(2)
                        traj_inputs = sgn_keypoints
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
                self.mixup(mixup_param, sgn_videos, sgn_heatmaps, labels, ip_c=sgn_videos_low, ip_d=sgn_heatmaps_low, 
                            low2high=self.transform_cfg.get('mixup_low2high', False), start_idx=kwargs.pop('start_idx', None), bag_labels=bag_labels)
            aug = kwargs.pop('aug', None)
            aug_shuffle = aug
            if aug is not None:
                aug_shuffle = aug[index]

        if self.input_streams == ['rgb']:
            s3d_outputs = self.visual_backbone(sgn_videos=sgn_videos)
        elif self.input_streams == ['keypoint']:
            s3d_outputs = self.visual_backbone_keypoint(sgn_videos=sgn_heatmaps)
        elif self.input_streams == ['trajectory']:
            s3d_outputs = self.traj_backbone(x=traj_inputs)
        elif len(self.input_streams)==2:
            s3d_outputs = self.visual_backbone_twostream(x_rgb=sgn_videos, x_pose=sgn_heatmaps)
            if 'trajectory' in self.input_streams:
                traj_outputs = self.traj_backbone(x=traj_inputs)
        elif len(self.input_streams)==4:
            s3d_outputs = self.visual_backbone_fourstream(sgn_videos, sgn_videos_low, sgn_heatmaps, sgn_heatmaps_low)

        if self.fuse_method is None:
            assert len(self.input_streams)==1, self.input_streams
            assert self.cfg['pyramid']['rgb'] == self.cfg['pyramid']['pose']
            if 'rgb' in self.input_streams:
                if 'vit' in self.cfg:
                    head_outputs = s3d_outputs
                else:
                    head_outputs = self.visual_head(x=s3d_outputs['sgn_feature'], labels=labels)

                # if self.cfg['aux_head']['num'] > 0:
                #     for i in range(len(self.visual_head_remain)):
                #         head_ops = self.visual_head_remain[i](x=s3d_outputs['fea_lst'][-1+i-self.cfg['aux_head']['num']])
                #         aux_logits['rgb'].append(head_ops['gloss_logits'])
            
            elif 'keypoint' in self.input_streams or 'keypoint_coord' in self.input_streams:
                if 'vit' in self.cfg:
                    head_outputs = s3d_outputs
                else:
                    head_outputs = self.visual_head_keypoint(x=s3d_outputs['sgn_feature'], labels=labels)
                # if self.cfg['aux_head']['num'] > 0:
                #     for i in range(len(self.visual_head_keypoint_remain)):
                #         head_ops = self.visual_head_keypoint_remain[i](x=s3d_outputs['fea_lst'][-1+i-self.cfg['aux_head']['num']])
                #         aux_logits['keypoint'].append(head_ops['gloss_logits'])

            elif 'trajectory' in self.input_streams:
                head_outputs = self.visual_head_traj(x=s3d_outputs['sgn_feature'], labels=labels)
            else:
                raise ValueError
            if self.contras_setting is not None:
                if 'late' in self.contras_setting:
                    head_outputs['word_ensemble_gloss_logits'] = (head_outputs['gloss_probabilities'] + head_outputs['word_fused_gloss_probabilities']).log()
                    head_outputs['word_ensemble_gloss_probabilities'] = head_outputs['word_ensemble_gloss_logits'].softmax(1)
                elif 'margin' in self.contras_setting:
                    head_outputs['word_ensemble_gloss_logits'] = (head_outputs['gloss_probabilities'] + head_outputs['word_margin_gloss_probabilities']).log()
                    head_outputs['word_ensemble_gloss_probabilities'] = head_outputs['word_ensemble_gloss_logits'].softmax(1)
            outputs = {**head_outputs}

        elif 'sephead' in self.fuse_method or 'triplehead' in self.fuse_method:
            assert len(self.input_streams)==2
            # rgb
            fbank_pose = kwargs.pop('fbank_pose', None)
            head_outputs_rgb = self.visual_head(x=s3d_outputs['rgb_fea_lst'][-1], labels=labels, fbank=fbank_pose, temp_idx=temp_idx, bag_labels=bag_labels)
            head_rgb_input = s3d_outputs['rgb_fea_lst'][-1]  #B,T,C
            
            # if self.cfg['aux_head']['num'] > 0:
            #     for i in range(len(self.visual_head_remain)):
            #         head_ops = self.visual_head_remain[i](x=s3d_outputs['rgb_fea_lst'][-1+i-self.cfg['aux_head']['num']])
            #         aux_logits['rgb'].append(head_ops['gloss_logits'])

            # keypoint
            fbank_rgb = kwargs.pop('fbank_rgb', None)
            head_keypoint_input = s3d_outputs['pose_fea_lst'][-1]
            head_outputs_keypoint = self.visual_head_keypoint(x=s3d_outputs['pose_fea_lst'][-1], labels=labels, fbank=fbank_rgb, temp_idx=temp_idx, bag_labels=bag_labels)
            
            # if self.cfg['aux_head']['num'] > 0:
            #     for i in range(len(self.visual_head_keypoint_remain)):
            #         head_ops = self.visual_head_keypoint_remain[i](x=s3d_outputs['pose_fea_lst'][-1+i-self.cfg['aux_head']['num']])
            #         aux_logits['keypoint'].append(head_ops['gloss_logits'])

            head_outputs = {'gloss_logits': None, 
                            'rgb_gloss_logits': head_outputs_rgb['gloss_logits'],
                            'keypoint_gloss_logits': head_outputs_keypoint['gloss_logits'],
                            'gloss_probabilities': None,
                            'rgb_gloss_probabilities': head_outputs_rgb['gloss_probabilities'],
                            'keypoint_gloss_probabilities': head_outputs_keypoint['gloss_probabilities'],
                            'rgb_split_logits': head_outputs_rgb['split_logits'],
                            'keypoint_split_logits': head_outputs_keypoint['split_logits'],
                            'rgb_bag_logits': head_outputs_rgb['bag_logits'],
                            'keypoint_bag_logits': head_outputs_keypoint['bag_logits'],
                            'rgb_cam': head_outputs_rgb['cam'],
                            'keypoint_cam': head_outputs_keypoint['cam'],
                            'head_rgb_input': head_rgb_input, 
                            'head_keypoint_input': head_keypoint_input
                            }
            
            if 'triplehead' in self.fuse_method:
                assert self.visual_head_fuse != None
                joint_mixup_param = self.cfg.get('joint_mixup', None)
                if 'plus' in self.fuse_method:
                    if joint_mixup_param:
                        if not mixup_param:
                            fused_sgn_features, _, _, _, fused_y_a, fused_y_b, fused_lam, _, _ = \
                                self.mixup(joint_mixup_param, head_rgb_input, head_keypoint_input, labels, cross=True)
                        else:
                            # do intra-/inter-modal mixup simultaneously
                            fused_sgn_features, _, _, _, fused_y_a, fused_y_b, fused_lam, index, _ = \
                                self.mixup(joint_mixup_param, head_rgb_input, head_keypoint_input, labels, cross=True, do_joint_mixup=not do_mixup)
                            fused_y_a, fused_y_b = y_a[index], y_b[index]
                    else:
                        fused_sgn_features = head_rgb_input + head_keypoint_input
                elif 'cat' in self.fuse_method:
                    fused_sgn_features = torch.cat([head_rgb_input, head_keypoint_input], dim=-1)
                else:
                    raise ValueError

                fbank_fuse = None
                # if fbank_rgb is not None:
                #     fbank_fuse = torch.cat([fbank_pose, fbank_rgb], dim=-1)
                head_outputs_fuse = self.visual_head_fuse(x=fused_sgn_features, labels=labels, fbank=fbank_fuse, temp_idx=temp_idx, bag_labels=bag_labels) 
                head_outputs['fuse_gloss_probabilities'] = head_outputs_fuse['gloss_probabilities']
                head_outputs['fuse_gloss_logits'] = head_outputs_fuse['gloss_logits']
                head_outputs['fuse_split_logits'] = head_outputs_fuse['split_logits']
                head_outputs['fuse_bag_logits'] = head_outputs_fuse['bag_logits']
                head_outputs['fuse_gloss_feature'] = head_outputs_fuse['gloss_feature']
                head_outputs['head_fuse_input'] = fused_sgn_features
                head_outputs['fuse_cam'] = head_outputs_fuse['cam']

                head_outputs['rgb_word_fused_gloss_logits'] = head_outputs_rgb['word_fused_gloss_logits']
                head_outputs['keypoint_word_fused_gloss_logits'] = head_outputs_keypoint['word_fused_gloss_logits']
                head_outputs['fuse_word_fused_gloss_logits'] = head_outputs_fuse['word_fused_gloss_logits']

                head_outputs['rgb_word_fused_gloss_probabilities'] = head_outputs_rgb['word_fused_gloss_probabilities']
                head_outputs['keypoint_word_fused_gloss_probabilities'] = head_outputs_keypoint['word_fused_gloss_probabilities']
                head_outputs['fuse_word_fused_gloss_probabilities'] = head_outputs_fuse['word_fused_gloss_probabilities']

                head_outputs['rgb_topk_idx'] = head_outputs_rgb['topk_idx']
                head_outputs['keypoint_topk_idx'] = head_outputs_keypoint['topk_idx']
                head_outputs['fuse_topk_idx'] = head_outputs_fuse['topk_idx']

                if self.cfg['visual_head'].get('weighted', False):
                    # print('weight')
                    head_prob_weight = self.head_prob_weight_learner([head_outputs_rgb['gloss_feature'], head_outputs_keypoint['gloss_feature'], head_outputs_fuse['gloss_feature']])
                    #[B,3]
                    head_prob_weight = head_prob_weight.split(1, dim=1)
                    head_outputs['ensemble_last_gloss_logits'] = (head_prob_weight[0]*head_outputs['rgb_gloss_probabilities']+\
                        head_prob_weight[1]*head_outputs['keypoint_gloss_probabilities'] + head_prob_weight[2]*head_outputs['fuse_gloss_probabilities']).log()
                    head_outputs['head_prob_rgb_weight'], head_outputs['head_prob_keypoint_weight'], head_outputs['head_prob_fuse_weight'] = head_prob_weight
                else:
                    if 'trajectory' in self.input_streams:
                        head_outputs['ensemble_last_gloss_logits'] = (head_outputs['fuse_gloss_probabilities']+\
                        head_outputs['rgb_gloss_probabilities']+head_outputs['keypoint_gloss_probabilities']+head_outputs['traj_gloss_probabilities']).log()
                    else:
                        head_outputs['ensemble_last_gloss_logits'] = (head_outputs['fuse_gloss_logits'].softmax(dim=-1)+\
                            head_outputs['rgb_gloss_logits'].softmax(dim=-1)+head_outputs['keypoint_gloss_logits'].softmax(-1)).log()
                    head_outputs['ensemble_last_gloss_raw_logits'] = head_outputs['ensemble_last_gloss_logits']
            
            elif 'sephead_logits_plus' in self.fuse_method: 
                assert 'loss2'  in self.fuse_method, self.fuse_method   
                if self.cfg.get('plus_type', 'prob')=='prob':
                    sum_probs = head_outputs['rgb_gloss_logits'].softmax(-1)+head_outputs['keypoint_gloss_logits'].softmax(-1)
                    head_outputs['ensemble_last_gloss_logits'] = sum_probs.log()
                # elif self.cfg.get('plus_type', 'prob')=='logits':
                #     head_outputs['ensemble_last_gloss_logits'] = head_outputs['rgb_gloss_logits']+head_outputs['keypoint_gloss_logits']
                else:
                    raise ValueError
            
            else:
                raise ValueError 
            
            head_outputs['ensemble_last_gloss_probabilities_log'] = head_outputs['ensemble_last_gloss_logits'].log_softmax(1) 
            head_outputs['ensemble_last_gloss_probabilities'] = head_outputs['ensemble_last_gloss_logits'].softmax(1)
            outputs = {**head_outputs}
        
        elif 'four' in self.fuse_method:
            joint_mixup_param = None
            outputs = {}
            keys_of_int = ['gloss_logits', 'word_fused_gloss_logits', 'topk_idx', 'word_emb_att_scores']
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

        if self.fuse_method == None or 'loss1' in self.fuse_method:
            if head_outputs['gloss_logits'] is not None:
                if mixup_param:
                    outputs['recognition_loss'] = lam*self.compute_recognition_loss(logits=head_outputs['gloss_logits'], labels=y_a, bag_loss=self.bag_loss_flags[0]) + \
                                                (1.-lam)*self.compute_recognition_loss(logits=head_outputs['gloss_logits'], labels=y_b, bag_loss=self.bag_loss_flags[0])
                    # print('mixup: ', lam)
                else:
                    outputs['recognition_loss'] = self.compute_recognition_loss(logits=head_outputs['gloss_logits'], labels=labels, bag_loss=self.bag_loss_flags[0])
        
        elif 'loss2' in self.fuse_method or 'loss3' in self.fuse_method or 'triplehead' in self.fuse_method or 'four' in self.fuse_method:
            assert len(self.input_streams)==2 or len(self.input_streams)==4
            if len(self.input_streams)==2:
                key_lst = ['rgb', 'keypoint', 'traj', 'fuse']
            else:
                key_lst = effect_head_lst
                head_outputs = outputs
            for k in key_lst:
                if f'{k}_gloss_logits' in head_outputs:
                    if k != 'fuse' or not self.cfg.get('joint_mixup', None):
                        if mixup_param:
                            # print('recloss: ', k)
                            outputs[f'recognition_loss_{k}'] = lam*self.compute_recognition_loss(
                                                                        logits=head_outputs[f'{k}_gloss_logits'], labels=y_a, 
                                                                        head_name=k,
                                                                        aug=aug,
                                                                        bag_labels=bag_labels,
                                                                        iou_labels=iou_labels,
                                                                        bag_loss=self.bag_loss_flags[0],
                                                                        bag_logits=head_outputs[f'{k}_bag_logits'][0]
                                                                        ) + \
                                                                (1.-lam)*self.compute_recognition_loss(
                                                                        logits=head_outputs[f'{k}_gloss_logits'], labels=y_b, 
                                                                        head_name=k,
                                                                        aug=aug_shuffle,
                                                                        bag_labels=bag_labels,
                                                                        iou_labels=iou_labels,
                                                                        bag_loss=self.bag_loss_flags[0],
                                                                        bag_logits=head_outputs[f'{k}_bag_logits'][0]
                                                                        )
                        else:
                            # print(k)
                            outputs[f'recognition_loss_{k}'] = self.compute_recognition_loss(
                                logits=head_outputs[f'{k}_gloss_logits'],
                                labels=labels,
                                bag_labels=bag_labels,
                                iou_labels=iou_labels,
                                bag_loss=self.bag_loss_flags[0],
                                bag_logits=head_outputs[f'{k}_bag_logits'][0]
                                )
                    else:
                        if mixup_param:
                            l_a = self.compute_recognition_loss(logits=head_outputs[f'{k}_gloss_logits'], labels=y_a)
                            l_b = self.compute_recognition_loss(logits=head_outputs[f'{k}_gloss_logits'], labels=y_b)
                            l_fused_a = self.compute_recognition_loss(logits=head_outputs[f'{k}_gloss_logits'], labels=fused_y_a)
                            l_fused_b = self.compute_recognition_loss(logits=head_outputs[f'{k}_gloss_logits'], labels=fused_y_b)
                            outputs[f'recognition_loss_{k}'] = fused_lam * (lam*l_a + (1.-lam)*l_b) + (1.-fused_lam) * (lam*l_fused_a + (1.-lam)*l_fused_b)
                        else:
                            # apply mixup to fuse features
                            outputs[f'recognition_loss_{k}'] = fused_lam*self.compute_recognition_loss(
                                logits=head_outputs[f'{k}_gloss_logits'],
                                labels=fused_y_a) + \
                                (1.-fused_lam)*self.compute_recognition_loss(
                                logits=head_outputs[f'{k}_gloss_logits'],
                                labels=fused_y_b)

            if 'loss2' in self.fuse_method:
                # print('loss2')
                outputs['recognition_loss'] = outputs['recognition_loss_rgb'] + outputs['recognition_loss_keypoint']

            elif 'triplehead' in self.fuse_method:
                # print('triplehead')
                if 'single_loss' in self.fuse_method:
                    # print("only impose one single loss over the average prob")
                    outputs['recognition_loss'] = 0.0*outputs['recognition_loss_rgb'] + 0.0*outputs['recognition_loss_keypoint'] + outputs['recognition_loss_fuse'] #outputs['recognition_loss_ensemble_last']
                else:
                    if 'trajectory' in self.input_streams:
                        outputs['recognition_loss'] = outputs['recognition_loss_rgb'] + outputs['recognition_loss_keypoint'] + outputs['recognition_loss_fuse'] + outputs['recognition_loss_traj']
                    else:
                        outputs['recognition_loss'] = outputs['recognition_loss_rgb'] + outputs['recognition_loss_keypoint'] + outputs['recognition_loss_fuse']
            
            elif 'four' in self.fuse_method:
                for k in effect_head_lst:
                    outputs['recognition_loss'] = outputs.get('recognition_loss', torch.tensor(0.0).to(sgn_videos.device))\
                                                + outputs[f'recognition_loss_{k}']

        else:
            raise ValueError
        outputs['total_loss'] = outputs['recognition_loss']

        if self.head_split_setting is not None and 'split' in self.head_split_setting and temp_idx is not None:
            for k in ['rgb_', 'keypoint_', 'fuse_']:
                if 'cam' in self.head_split_setting or 'att' in self.head_split_setting:
                    cam = head_outputs[k+'cam']
                    if cam is not None:
                        if 'cam' in self.head_split_setting:
                            cam = F.log_softmax(cam, dim=-1)
                        else:
                            cam = cam.squeeze(1).log()
                        B, T = cam.shape
                        cam_label = torch.zeros_like(cam).detach()
                        # make label
                        if 'uniform' in self.head_split_setting:
                            i = 0
                            for s, e in temp_idx:
                                s, e = s.item(), e.item()
                                cam_label[i, s:e] = 1.0/(e-s)
                                i += 1
                        elif 'gaussian' in self.head_split_setting:
                            i = 0
                            idx = torch.arange(T).to(cam.device)
                            ratio = 4 if 'gaussian_4' in self.head_split_setting else 2
                            for s, e in temp_idx:
                                s, e = s.item(), e.item()
                                mu, sigma = (s+e)/2, (e-s)/ratio
                                cam_label[i] = torch.exp(-(idx-mu)**2 / (2*sigma**2))
                                i += 1
                            cam_label = F.normalize(cam_label, p=1.0, dim=-1)
                        cam_loss = self.kl_loss_func(cam, cam_label)
                        # print('cam loss: ', cam_loss)
                        outputs[k+'cam_loss'] = cam_loss
                        outputs['total_loss'] = outputs['total_loss'] + cam_loss
                
                split_logits = head_outputs[k+'split_logits']
                if len(split_logits) > 0:
                    blank_label = torch.zeros_like(labels).long()
                    if 'att' in self.head_split_setting:
                        split_loss = self.compute_recognition_loss(logits=split_logits[0], labels=labels, bag_labels=bag_labels)
                    elif 'nonblk' in self.head_split_setting:
                        # print('nonblk_'+k)
                        split_loss = self.compute_recognition_loss(logits=split_logits[0], labels=labels, bag_labels=bag_labels, 
                                                                   bag_loss=self.bag_loss_flags[1], bag_logits=head_outputs[k+'bag_logits'][1])
                    else:
                        split_loss = self.compute_recognition_loss(logits=split_logits[0], labels=blank_label, bag_labels=bag_labels)+\
                                        self.compute_recognition_loss(logits=split_logits[2], labels=blank_label, bag_labels=bag_labels)+\
                                        self.compute_recognition_loss(logits=split_logits[1], labels=labels, bag_labels=bag_labels)
                    outputs[k+'split_loss'] = split_loss
                    # print('split_loss: ', split_loss)
                    outputs['total_loss'] = outputs['total_loss'] + split_loss

        if self.contras_setting is not None:
            if len(self.input_streams) == 2:
                key_lst = ['rgb_', 'keypoint_', 'fuse_']
            elif len(self.input_streams) == 3:
                key_lst = ['rgb_', 'keypoint_', 'traj_', 'fuse_']
            elif len(self.input_streams) == 4:
                key_lst = [e+'_' for e in effect_head_lst]
            else:
                key_lst = ['']
            for k in key_lst:
                contras_loss = contras_loss2 = torch.tensor(0.0).to(sgn_videos.device) if sgn_videos is not None else torch.tensor(0.0).to(sgn_keypoints.device)
                if 'dual' in self.contras_setting:
                    if 'margin' in self.contras_setting:
                        word_margin_gloss_logits = outputs[f'{k}word_margin_gloss_logits']
                        contras_loss = self.contras_loss_func(word_margin_gloss_logits, labels)
                    # elif 'multi_label' in self.contras_setting:
                    #     word_fused_gloss_logits = outputs[f'{k}word_fused_gloss_logits']
                    #     contras_loss = self.contras_loss_func(word_fused_gloss_logits, labels)
                    else:
                        word_fused_gloss_logits = outputs[f'{k}word_fused_gloss_logits']
                        if word_fused_gloss_logits is None:
                            # only apply to some heads
                            continue
                        topk_idx = outputs[f'{k}topk_idx']
                        if k != 'fuse_':
                            if mixup_param is not None:
                                # print('word_fused: ', k)
                                contras_loss = self.contras_loss_func(word_fused_gloss_logits, labels, topk_idx=topk_idx, mixup_lam=lam, y_a=y_a, y_b=y_b)
                                if 'xmodal' in self.contras_setting and fbank_rgb is not None:
                                    xmodal_fused_gloss_logits = outputs[f'{k}xmodal_fused_gloss_logits']
                                    contras_loss2 = self.contras_loss_func(xmodal_fused_gloss_logits, labels, topk_idx=topk_idx, mixup_lam=lam, y_a=y_a, y_b=y_b)
                            else:
                                mixup_lam = None  #outputs[f'{k}mixup_lam']
                                contras_loss = self.contras_loss_func(word_fused_gloss_logits, labels, topk_idx=topk_idx, mixup_lam=mixup_lam)
                        else:
                            if mixup_param and joint_mixup_param:
                                contras_loss = self.contras_loss_func(word_fused_gloss_logits, labels, topk_idx=topk_idx, mixup_lam=lam, y_a=y_a, y_b=y_b,
                                                                        fused_mixup_lam=fused_lam, fused_y_a=fused_y_a, fused_y_b=fused_y_b)
                            elif mixup_param and not joint_mixup_param:
                                # print('word_fused: ', k)
                                contras_loss = self.contras_loss_func(word_fused_gloss_logits, labels, topk_idx=topk_idx, mixup_lam=lam, y_a=y_a, y_b=y_b)
                                if 'xmodal' in self.contras_setting and fbank_fuse is not None:
                                    xmodal_fused_gloss_logits = outputs[f'{k}xmodal_fused_gloss_logits']
                                    contras_loss2 = self.contras_loss_func(xmodal_fused_gloss_logits, labels, topk_idx=topk_idx, mixup_lam=lam, y_a=y_a, y_b=y_b)
                            elif not mixup_lam and joint_mixup_param:
                                contras_loss = self.contras_loss_func(word_fused_gloss_logits, labels, topk_idx=topk_idx, mixup_lam=fused_lam, y_a=fused_y_a, y_b=fused_y_b)
                            else:
                                contras_loss = self.contras_loss_func(word_fused_gloss_logits, labels, topk_idx=topk_idx)
                
                if 'contras' in self.contras_setting:
                    scores = outputs[f'{k}word_emb_att_scores']
                    contras_loss2 = lam*self.contras_loss_func2(scores, y_a) + \
                                        (1.-lam)*self.contras_loss_func2(scores, y_b)
                
                outputs[f'contras_loss_{k}'] = contras_loss
                outputs[f'contras_loss2_{k}'] = contras_loss2

                #contras only "=" other wise "+="
                if 'only' in self.contras_setting:
                    outputs['total_loss'] = self.contras_loss_weight * contras_loss
                elif 'late' in self.contras_setting:
                    ep1, ep2, contras_loss_weight = self.contras_loss_weight.split('_')
                    ep1, ep2, contras_loss_weight = int(ep1), int(ep2), float(contras_loss_weight)
                    if epoch < ep1:
                        cur_loss_weight = 0.0
                    elif epoch >= ep1 and epoch < ep2:
                        cur_loss_weight = contras_loss_weight*(epoch-ep1)/(ep2-ep1)
                    else:
                        cur_loss_weight = contras_loss_weight
                    outputs['total_loss'] = outputs['total_loss'] + cur_loss_weight * contras_loss
                elif 'cosine' in self.contras_setting:
                    cur_loss_weight = 0.5 * self.contras_loss_weight * (1.0 + math.cos(epoch*math.pi/100))
                    outputs['total_loss'] = outputs['total_loss'] + cur_loss_weight * contras_loss
                    if 'xmodal' in self.contras_setting:
                        start = self.cfg['visual_head']['fbank']['start']
                        if epoch >= start:
                            cur_loss_weight = 0.5 * self.contras_loss_weight * (1.0 + math.cos((epoch-start)*math.pi/(100-start)))
                            outputs['total_loss'] = outputs['total_loss'] + cur_loss_weight * contras_loss2
                        # print('decay')
                else:
                    outputs['total_loss'] = outputs['total_loss'] + self.contras_loss_weight * contras_loss
                outputs['total_loss'] = outputs['total_loss'] + contras_loss2

                if f'{k}word_emb_score_lst' in s3d_outputs:
                    extra_scores = s3d_outputs[f'{k}word_emb_score_lst']
                    # print(len(extra_scores))
                    for i in range(len(extra_scores)):
                        contras_loss = self.contras_loss_func(scores.permute(0,4,1,2,3) if scores.ndim>2 else scores, 
                                                    labels.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(scores[...,0]) if scores.ndim>2 else labels)
                        outputs[f'contras_loss_{k}{str(i)}'] = contras_loss
                        #contras only "=" other wise "+="
                        outputs['total_loss'] = outputs['total_loss'] + self.contras_loss_weight * contras_loss
        
        if 'cross_distillation' in self.cfg:
            soft_or_hard = self.cfg['cross_distillation'].get('hard_or_soft','soft')
            assert soft_or_hard in ['soft', 'hard']
            assert self.fuse_method in ['sephead_logits_plus_loss2', 'triplehead_cat_bilateral']
            if soft_or_hard == 'soft':
                loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            else:
                loss_func = torch.nn.CrossEntropyLoss(reduction='mean') #divided by batch_size
            
            if type(self.cfg['cross_distillation']['types'])==list:
                self.cfg['cross_distillation']['types']={t:self.cfg['cross_distillation'].get('loss_weight',1) 
                    for t in self.cfg['cross_distillation']['types']}
            
            for teaching_type, loss_weight in self.cfg['cross_distillation']['types'].items():
                teacher = teaching_type.split('_teaches_')[0]
                student = teaching_type.split('_teaches_')[1]
                # print('teacher--student: ', teacher, student)
                assert teacher in ['rgb', 'keypoint', 'ensemble_last', 'fuse', 'ensemble_early'], teacher
                assert student in ['rgb', 'keypoint','fuse', 'auxes']
                if soft_or_hard == 'soft':
                    teacher_prob = outputs[f'{teacher}_gloss_probabilities']
                else:
                    teacher_prob = torch.argmax(outputs[f'{teacher}_gloss_probabilities'], dim=-1) #B,T,
                
                if self.cfg['cross_distillation']['teacher_detach'] == True:
                    teacher_prob = teacher_prob.detach()
                if soft_or_hard == 'soft':
                    student_log_prob = outputs[f'{student}_gloss_probabilities'].log()
                    outputs[f'{teaching_type}_loss'] = loss_func(input=student_log_prob, target=teacher_prob)
                else:
                    student_logits = outputs[f'{student}_gloss_logits']
                    B, T, V = student_logits.shape
                    outputs[f'{teaching_type}_loss'] = loss_func(input=student_logits.view(-1, V), target=teacher_prob.view(-1))
                outputs['recognition_loss'] = outputs['recognition_loss'] + outputs[f'{teaching_type}_loss'] * loss_weight

        return outputs


    def forward(self, is_train, labels, sgn_videos=None, sgn_keypoints=None, epoch=0, **kwargs):
        if len(sgn_videos) == 1:
            # print('video shape: ', sgn_videos[0].shape)
            return self._forward_impl(is_train, labels, sgn_videos=sgn_videos[0], sgn_keypoints=sgn_keypoints[0], epoch=epoch, **kwargs)

        else:
            if len(self.input_streams) == 1:
                if sgn_keypoints is None:
                    sgn_keypoints = [None] * len(sgn_videos)
                outputs = {}
                for s_videos, s_keypoints in zip(sgn_videos, sgn_keypoints):
                    n_frames = s_videos.shape[1]
                    # print(n_frames)
                    temp_ops = self._forward_impl(is_train, labels, sgn_videos=s_videos, sgn_keypoints=s_keypoints, epoch=epoch, **kwargs)
                    for k, v in temp_ops.items():
                        outputs[str(n_frames)+'_'+k] = v
                        if 'total_loss' in k:
                            outputs['total_loss'] = outputs.get('total_loss', torch.tensor(0.0).to(v.device)) + v
                        if k=='gloss_logits':
                            outputs['frame_ensemble_gloss_probabilities'] = outputs.get('frame_ensemble_gloss_probabilities', torch.zeros_like(v).to(v.device)) + v.softmax(-1)
                outputs['total_loss'] = outputs['total_loss'] / len(sgn_videos)
                outputs['frame_ensemble_gloss_probabilities'] = outputs['frame_ensemble_gloss_probabilities'] / len(sgn_videos)
                outputs['frame_ensemble_gloss_logits'] = outputs['frame_ensemble_gloss_probabilities'].log()
                return outputs
            
            elif len(self.input_streams) == 4:
                # print(sgn_videos[0].shape, sgn_videos[1].shape, sgn_keypoints[0].shape, sgn_keypoints[1].shape)
                return self._forward_impl(is_train, labels, sgn_videos=sgn_videos[0], sgn_keypoints=sgn_keypoints[0], epoch=epoch,
                                        sgn_videos_low=sgn_videos[1], sgn_keypoints_low=sgn_keypoints[1], **kwargs)
            
            elif len(self.input_streams) == 2:
                sgn_videos, sgn_keypoints = sgn_videos
                # print('video1 shape: ', sgn_videos.shape)
                # print('video2 shape: ', sgn_keypoints.shape)
                return self._forward_impl(is_train, labels, sgn_videos=sgn_videos, sgn_keypoints=sgn_keypoints, epoch=epoch, **kwargs)