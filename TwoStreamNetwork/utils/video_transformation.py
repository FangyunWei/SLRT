# define project dependency
from numpy.random import SeedSequence
from functools import partial
import os, glob, pickle, random
import numpy as np
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.utils import data 
import torchvision
from torchvision import transforms
import torch.distributed as dist
import torchvision.utils as vutils
import torch.nn.functional as F
import sys
import utils.augmentation as A
import utils.transforms as T

def crop(vid, i, j, h, w):
    return vid[..., i:(i + h), j:(j + w)]

def center_crop(vid, output_size):
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(vid, i, j, th, tw)

class ChannelSwap:
    def __init__(self):
        kk = 0
    def __call__(self, tensor_4d):
        # [RGB,t,h,w] -> [BGR,t,h,w]
        return tensor_4d[[2,1,0],:,:,:]
def resize(images, size):
    return torch.nn.functional.interpolate(
        images,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
class AugmentOp:
    """
    Apply for video.
    """
    def __init__(self, aug_fn, *args, **kwargs):
        self.aug_fn = aug_fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, images):
        return self.aug_fn(images, *self.args, **self.kwargs)

def get_data_transform(mode, dataset_info):
    ## preprocess data (PIL-image list) before batch binding
    if mode == 'train':
        ops = [torchvision.transforms.RandomResizedCrop(
            size=224, 
            scale=(dataset_info.get('bottom_area',0.2), 1.0), 
            ratio=(dataset_info.get('aspect_ratio_min',3./4), 
                   dataset_info.get('aspect_ratio_max',4./3)))]
        # ops = [A.RandomSizedCrop(size=224, 
        #     consistent=True, 
        #     bottom_area=dataset_info.get('bottom_area',0.2),
        #     aspect_ratio_min=dataset_info.get('aspect_ratio_min',3./4),
        #     aspect_ratio_max=dataset_info.get('aspect_ratio_max',4./3),
        #     p=dataset_info.get('randomcrop_threshold',1),
        #     center_crop_size=dataset_info.get('center_crop_size',224),
        #     center_crop=dataset_info.get('center_crop',True))]
        if dataset_info['aug_hflip']:
            raise ValueError
            ops.append(A.RandomHorizontalFlip())
        #ops.append(A.Scale(dataset_info['img_size']))
        if dataset_info['color_jitter']:
            raise ValueError
            ops.append(A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.3, consistent=True))
    elif mode == 'val' or mode == 'test':
        ops = []
        if dataset_info.get('center_crop',True)==True:
            center_crop_size = dataset_info.get('center_crop_size', 224)
            ops.append(torchvision.transforms.CenterCrop(size=center_crop_size))
    else:
        raise NotImplementedError
    if dataset_info.get('network','s3d')=='s3d':
        ops.extend([
            #A.ToTensor(),
            #T.Stack(dim=1),
            AugmentOp(resize, **{'size': dataset_info['img_size']}),
            T.Normalize_all_channel(mean=0.5, std=0.5, channel=0),
            #ChannelSwap(), out side
        ])
    elif dataset_info.get('network','s3d')=='resnet':
        ops.extend([
            A.ToTensor(),
            T.Stack(dim=1),
            AugmentOp(resize, **{'size': dataset_info['img_size']}),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=0)
        ])        
    data_transform = transforms.Compose(ops)
    return data_transform

def get_data_transform_oldscale(mode, dataset_info):
    ## preprocess data (PIL-image list) before batch binding
    if mode == 'train':
        ops = [A.RandomSizedCrop(size=224, 
            consistent=True, 
            bottom_area=dataset_info.get('bottom_area',0.2),
            aspect_ratio_min=dataset_info.get('aspect_ratio_min',3./4),
            aspect_ratio_max=dataset_info.get('aspect_ratio_max',4./3),
            p=dataset_info.get('randomcrop_threshold',1),
            center_crop_size=dataset_info.get('center_crop_size',224),
            center_crop=dataset_info.get('center_crop',True))]
        if dataset_info['aug_hflip']:
            ops.append(A.RandomHorizontalFlip())
        ops.append(A.Scale(dataset_info['img_size']))
        if dataset_info['color_jitter']:
            ops.append(A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.3, consistent=True))
    elif mode == 'val' or mode == 'test':
        ops = []
        if dataset_info.get('center_crop',True)==True:
            center_crop_size = dataset_info.get('center_crop_size', 224)
            ops.append(A.CenterCrop(size=center_crop_size, consistent=True))
        #ops.append(A.Scale(dataset_info['img_size']))
    else:
        raise NotImplementedError
    if dataset_info.get('network','s3d')=='s3d':
        ops.extend([
            A.ToTensor(),
            T.Stack(dim=1),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], channel=0),
            ChannelSwap(),
        ])
    elif dataset_info.get('network','s3d')=='resnet':
        ops.extend([
            A.ToTensor(),
            T.Stack(dim=1),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=0)
        ])        
    data_transform = transforms.Compose(ops)
    return data_transform

