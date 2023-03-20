import os, numpy as np, random, io
from PIL import Image
from utils.zipreader import ZipReader
import torch, torchvision


def get_selected_indexs(vlen, tmin=1, tmax=1, num_tokens=1, max_num_frames=400):
    if tmin==1 and tmax==1:
        if vlen <= max_num_frames:
            frame_index = np.arange(vlen)
            valid_len = vlen
        else:
            sequence = np.arange(vlen)
            an = (vlen - max_num_frames)//2
            en = vlen - max_num_frames - an
            frame_index = sequence[an: -en]
            valid_len = max_num_frames
        
        if (valid_len % 4) != 0:
            valid_len -= (valid_len % 4)
            frame_index = frame_index[:valid_len]

        assert len(frame_index) == valid_len, (frame_index, valid_len)
        return frame_index, valid_len
    
    min_len = int(tmin*vlen)
    max_len = min(max_num_frames, int(tmax*vlen))
    selected_len = np.random.randint(min_len, max_len+1)
    if (selected_len%4) != 0:
        selected_len += (4-(selected_len%4))
    if selected_len<=vlen: 
        selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
    else: 
        copied_index = np.random.randint(0,vlen,selected_len-vlen)
        selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))

    if selected_len <= max_num_frames:
        frame_index = selected_index
        valid_len = selected_len
    else:
        assert False, (vlen, selected_len, min_len, max_len)
    assert len(frame_index) == valid_len, (frame_index, valid_len)
    return frame_index, valid_len

def read_img(path, dataset_name, csl_cut, csl_resize=-1):
    zip_data = ZipReader.read(path)
    rgb_im = Image.open(io.BytesIO(zip_data)).convert('RGB')    
    if dataset_name.lower() == 'csl-daily': 
        if csl_cut:
            rgb_im = rgb_im.crop((0,80,512,512))
        if csl_resize!=-1:
            if csl_cut:
                assert csl_resize==[320,270] 
            else:
                assert csl_resize[0]==csl_resize[1]
            rgb_im = rgb_im.resize((csl_resize[0], csl_resize[1]))
    return rgb_im



def pil_list_to_tensor(pil_list, int2float=True):
    func = torchvision.transforms.PILToTensor()
    tensors = [func(pil_img) for pil_img in pil_list]
    tensors = torch.stack(tensors, dim=0)
    if int2float:
        tensors = tensors/255
    return tensors # (T,C,H,W)

def load_batch_video(zip_file, names, num_frames, transform_cfg, dataset_name, is_train, 
        pad_length='pad_to_max', pad='replicate',
        name2keypoint=None):
    if name2keypoint!=None:
        assert pad=='replicate', 'only support pad=replicate mode for keypoints'
    sgn_videos, sgn_keypoints = [], [] # (B,C,T,H,W)
    sgn_lengths = [] 
    for name, num in zip(names, num_frames):
        video, len_, selected_indexs = load_video(zip_file, name, num, transform_cfg, dataset_name, is_train)
        sgn_lengths.append(len_)
        sgn_videos.append(video) 
        if name2keypoint!=None:
            sgn_keypoints.append(name2keypoint[name][selected_indexs,:,:]) 
        else:
            sgn_keypoints.append(None)
    if pad_length=='pad_to_max':
        max_length = max(sgn_lengths)
    else:
        max_length = int(pad_length)

    padded_sgn_videos, padded_sgn_keypoints = [], []

    for video, keypoints, len_ in zip(sgn_videos, sgn_keypoints, sgn_lengths):
        video = pil_list_to_tensor(video, int2float=True) # (T,C,H,W)
        if len_<max_length:
            if pad=='zero':
                padding = torch.zeros_like(video[0:1]) 
            elif pad=='replicate':
                padding = video[-1,:,:,:].unsqueeze(0) 
            else:
                raise ValueError
            padding = torch.tile(padding, [max_length-len_, 1, 1, 1]) 
            padded_video = torch.cat([video, padding], dim=0)
            padded_sgn_videos.append(padded_video)     
        else:
            padded_sgn_videos.append(video)
        if name2keypoint!=None:
            if len_<max_length:
                padding = keypoints[-1].unsqueeze(0) 
                padding = torch.tile(padding, [max_length-len_, 1, 1]) 
                padded_keypoint = torch.cat([keypoints, padding], dim=0) 
                padded_sgn_keypoints.append(padded_keypoint)                   
            else:
                padded_sgn_keypoints.append(keypoints) 

    sgn_lengths = torch.tensor(sgn_lengths, dtype=torch.long)
    sgn_videos = torch.stack(padded_sgn_videos, dim=0) 
    if name2keypoint!=None:
        sgn_keypoints = torch.stack(padded_sgn_keypoints, dim=0) 
    else:
        sgn_keypoints = None
    return sgn_videos, sgn_keypoints, sgn_lengths




def load_video(zip_file, name, num_frames, transform_cfg, dataset_name, is_train):
    if 'temporal_augmentation' in transform_cfg  and is_train:
        tmin, tmax = transform_cfg['temporal_augmentation']['tmin'], transform_cfg['temporal_augmentation']['tmax']
    else:
        tmin, tmax = 1, 1
    if dataset_name.lower() in ['csl-daily', 'phoenix-2014t', 'phoenix-2014']: 
        if dataset_name.lower()=='csl-daily':
            image_path_list = ['{}@sentence_frames-512x512/{}/{:06d}.jpg'.format(zip_file, name, fi)
                for fi in range(num_frames)]
        elif dataset_name.lower()=='phoenix-2014t':
            image_path_list = ['{}@images/{}/images{:04d}.png'.format(zip_file, name, fi)
                for fi in range(1,num_frames+1)]
        elif dataset_name.lower()=='phoenix-2014':
            image_path_list = ['{}@{}.avi_pid0_fn{:06d}-0.png'.format(zip_file, name, fi)
                for fi in range(num_frames)]
        else:
            raise ValueError  
        selected_indexs, valid_len = get_selected_indexs(len(image_path_list), tmin=tmin, tmax=tmax)
        sequence = [read_img(image_path_list[i],dataset_name, 
                csl_cut=transform_cfg.get('csl_cut',True),
                csl_resize=transform_cfg.get('csl_resize',[320,320])) for i in selected_indexs]
        return sequence, valid_len, selected_indexs  
    else:
        raise ValueError        
