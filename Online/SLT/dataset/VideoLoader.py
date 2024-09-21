import os, numpy as np
from utils.gen_gaussian import gen_gaussian_hmap_op
from utils.video_transformation import get_data_transform
from utils.zipreader import ZipReader
from utils.misc import sliding_windows
import utils.augmentation as A
import io, torch, torchvision
from PIL import Image
import lintel, random

def _load_frame_nums_to_4darray(video, frame_nums):
    """Decodes a specific set of frames from `video` to a 4D numpy array.
    
    Args:
        video: Encoded video.
        dataset: Dataset meta-info, e.g., width and height.
        frame_nums: Indices of specific frame indices to decode, e.g.,
            [1, 10, 30, 35] will return four frames: the first, 10th, 30th and
            35 frames in `video`. Indices must be in strictly increasing order.

    Returns:
        A numpy array, loaded from the byte array returned by
        `lintel.loadvid_frame_nums`, containing the specified frames, decoded.
    """
    decoded_frames, width, height = lintel.loadvid_frame_nums(video,
                                               frame_nums=frame_nums)
    decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
    decoded_frames = np.reshape(
        decoded_frames,
        newshape=(-1, height, width, 3))

    return decoded_frames

def get_selected_indexs(vlen, tmin=1, tmax=1, num_tokens=1, max_num_frames=400):
    #num_tokens is ignored when level is set as 'sequence'
    if tmin==1 and tmax==1:
        #output deterministic results
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
    min_len = min(min_len, max_len)
    assert max_len+1>min_len, (min_len, max_len+1)
    selected_len = np.random.randint(min_len, max_len+1)
    if (selected_len%4) != 0:
        selected_len += (4-(selected_len%4))
    if selected_len<=vlen: #speed up
        selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
    else: #slow down boring video we need to copy some frames
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
    if dataset_name.lower() in ['csl','cslr']: #cslr won'r enter here
        if csl_cut:
            rgb_im = rgb_im.crop((0,80,512,512))
        if csl_resize!=-1:
            if csl_cut:
                assert csl_resize==[320,270] #width height
            else:
                assert csl_resize[0]==csl_resize[1]
            rgb_im = rgb_im.resize((csl_resize[0], csl_resize[1]))
    return rgb_im



def pil_list_to_tensor(pil_list, int2float=True):
    func = torchvision.transforms.PILToTensor()
    tensors = [func(pil_img) for pil_img in pil_list]
    #list of C H W
    tensors = torch.stack(tensors, dim=0)
    if int2float:
        tensors = tensors/255
    return tensors #T,C,H,W

def generate_pseudo(memory_bank, gloss_length_distribution, sample='class_imbalance', min_length=2, max_length=8):
    #memory_bank gls[]
    length = np.random.normal(gloss_length_distribution[0], gloss_length_distribution[1])
    length = np.clip(int(length), min_length, max_length)
    sequence, pseudo_label = [], [] 
    sorted_gls = sorted(memory_bank, key=lambda g:len(memory_bank[g])*-1) # descending order
    if sample=='class_balance':
        gls_i = np.random.randint(low=0, high=len(sorted_gls), size=[length])
        for i in gls_i: 
            pseudo_label.append(sorted_gls[i])
            j = np.random.randint(low=0, high=len(memory_bank[sorted_gls[i]]))
            sequence.append(memory_bank[sorted_gls[i]][j])
    elif sample=='class_imbalance':
        instances = [] #[gls, ins]
        for gls, ls in memory_bank.items():
            instances.extend([[gls, l] for l in ls])
        ins_i = np.random.randint(low=0, high=len(instances), size=[length])
        for i in ins_i:
            pseudo_label.append(instances[i][0])
            sequence.append(instances[i][1])
    else:
        raise ValueError
    name_sequence = [s[0] for s in sequence]
    boundary_sequence = [s[1] for s in sequence]
    return name_sequence, boundary_sequence, pseudo_label #[[name,[s,e]],], label


def load_batch_video(zip_file, names, num_frames, transform_cfg, dataset_name, is_train, 
        pad_length='pad_to_max', pad='replicate',
        name2keypoint=None, memory_bank=None, 
        gloss_length_distribution=None, 
        name_sequences=[None], boundary_sequences=[None],
        pseudo_cfg={}):
    #load_video and then pad
    if name2keypoint!=None:
        assert pad=='replicate', 'only support pad=replicate mode when input keypoints'
    sgn_videos, sgn_keypoints = [], [] # B,C T,H,W
    sgn_lengths = [] # B
    sgn_selected_indexs = []
    pseudo_outputs = {'gloss':[]}
    for ii, (name, num) in enumerate(zip(names, num_frames)):
        if 'pseudo' in name:
            if name_sequences[ii]==None:
                name_sequence, boundary_sequence, pseudo_gloss = generate_pseudo(
                    memory_bank=memory_bank, gloss_length_distribution=gloss_length_distribution,
                    **pseudo_cfg)
                pseudo_outputs['gloss'].append(pseudo_gloss)
            else:
                name_sequence = name_sequences[ii]
                boundary_sequence = boundary_sequences[ii]
                #print(name_sequence, boundary_sequence)
        else:
            name_sequence = [name]
            boundary_sequence = [(0, num-1)]
        video, len_, selected_indexs, pseudo_outputs_ = load_video(
            zip_file=zip_file, name=name, num_frames=num,
            name_sequence=name_sequence, boundary_sequence=boundary_sequence,
            transform_cfg=transform_cfg, dataset_name=dataset_name, 
            is_pseudo=('pseudo' in name), is_train=is_train)
        sgn_lengths.append(len_)
        sgn_videos.append(video) #wo transformed!!
        if name2keypoint!=None:
            if 'pseudo' in name:
                sgn_keypoint  = []
                for name, ind in zip(pseudo_outputs_['name_sequence'], pseudo_outputs_['selected_indexs']):
                    sgn_keypoint.append(name2keypoint[name][ind,:,:]) #N,D
                sgn_keypoint = np.concatenate(sgn_keypoint, axis=0)
                sgn_keypoints.append(sgn_keypoint)
            else:
                sgn_keypoints.append(name2keypoint[name][selected_indexs,:,:]) # T, N, D
        else:
            sgn_keypoints.append(None)
        sgn_selected_indexs.append(selected_indexs)

    if pad_length=='pad_to_max':
        max_length = max(sgn_lengths)
    else:
        max_length = int(pad_length)

    padded_sgn_videos, padded_sgn_keypoints = [], []

    for video, keypoints, len_ in zip(sgn_videos, sgn_keypoints, sgn_lengths):
        video = pil_list_to_tensor(video, int2float=True) #T,C,H,W
        if len_<max_length:
            if pad=='zero':
                padding = torch.zeros_like(video[0:1]) #1, C, h,W
            elif pad=='replicate':
                padding = video[-1,:,:,:].unsqueeze(0) #1, C, H, W
            else:
                raise ValueError
            padding = torch.tile(padding, [max_length-len_, 1, 1, 1]) #t, c, h, w
            padded_video = torch.cat([video, padding], dim=0)
            padded_sgn_videos.append(padded_video)     
        else:
            padded_sgn_videos.append(video)
        #keypoints
        if name2keypoint!=None:
            keypoints = torch.tensor(keypoints)
            if len_<max_length:
                padding = keypoints[-1].unsqueeze(0) #1,N,2(or 3)
                padding = torch.tile(padding, [max_length-len_, 1, 1]) #t, N, 2
                padded_keypoint = torch.cat([keypoints, padding], dim=0) 
                padded_sgn_keypoints.append(padded_keypoint)                   
            else:
                padded_sgn_keypoints.append(keypoints) #[T',N,2]

    sgn_lengths = torch.tensor(sgn_lengths, dtype=torch.long)
    sgn_videos = torch.stack(padded_sgn_videos, dim=0) #B,T,C,H,W
    if name2keypoint!=None:
        sgn_keypoints = torch.stack(padded_sgn_keypoints, dim=0) #B,T,N,2
    else:
        sgn_keypoints = None
    
    # sliding windows
    # sgn_videos, sgn_keypoints = sliding_windows(sgn_videos, sgn_keypoints, win_size=16, stride=8)
    # max_num = 20000
    # cur_num = sgn_videos.shape[0]
    # idx = [_ for _ in range(cur_num)]
    # if is_train:
    #     random.shuffle(idx)
    #     idx = sorted(idx[:max_num])
    # sgn_videos, sgn_keypoints = sgn_videos[idx], sgn_keypoints[idx]
    
    return sgn_videos, sgn_keypoints, sgn_lengths, sgn_selected_indexs, pseudo_outputs



datasetname2format = {
    'csl': '{}@sentence_frames-512x512/{}/{:06d}.jpg',
    'phoenix': '{}@images/{}/images{:04d}.png',
    'phoenix2014tsi': '{}@images/{}/images{:04d}.png',
    'phoenix2014': '{}@{}.avi_pid0_fn{:06d}-0.png',
    'phoenix2014si': '{}@{}.avi_pid0_fn{:06d}-0.png',
    'wlasl2000': '{}@WLASL2000/{}.mp4',
    'tvb': '{}@tvb/grouped/sign/{}/{:06d}.jpg'
}
def load_video(zip_file, name, num_frames, name_sequence, boundary_sequence, transform_cfg, dataset_name, is_pseudo, is_train):
    #temporal augmentation (train/test)
    if 'temporal_augmentation' in transform_cfg and is_train:
        tmin, tmax = transform_cfg['temporal_augmentation']['tmin'], transform_cfg['temporal_augmentation']['tmax']
    else:
        tmin, tmax = 1, 1
    if dataset_name.lower() in ['csl', 'phoenix', 'phoenix2014tsi', 'phoenix2014', 'phoenix2014si', 'tvb', 'phoenixcomb']: #read 2D image
        if dataset_name.lower() == 'phoenixcomb':
            if 'fullFrame' in name:
                path_format = datasetname2format['phoenix2014']
                dataset_name = 'phoenix2014'
            else:
                path_format = datasetname2format['phoenix']
                dataset_name = 'phoenix'
        else:
            path_format = datasetname2format[dataset_name.lower()]
        image_path_list = []
        for name, (start, end) in  zip(name_sequence, boundary_sequence):
            if dataset_name.lower() in ['phoenix', 'phoenix2014tsi']:
                start, end = start+1, end+1
            elif dataset_name.lower() == 'tvb':
                st_en = name.split('/')[-1]
                start = int(st_en.split('-')[0])
                end = start + num_frames - 1
            image_path_list.extend([path_format.format(zip_file, name, fi) 
                for fi in range(start, end+1)])#end inclusive
        # if dataset_name.lower()=='csl':
        #     image_path_list = ['{}@sentence_frames-512x512/{}/{:06d}.jpg'.format(zip_file, name, fi)
        #         for fi in range(num_frames)]
        # elif dataset_name.lower()=='phoenix':
        #     image_path_list = ['{}@images/{}/images{:04d}.png'.format(zip_file, name, fi)
        #         for fi in range(1,num_frames+1)]
        # elif dataset_name.lower()=='phoenix2014':
        #     image_path_list = ['{}@{}.avi_pid0_fn{:06d}-0.png'.format(zip_file, name, fi)
        #         for fi in range(num_frames)]
        # else:
        #     raise ValueError  
        selected_indexs, valid_len = get_selected_indexs(len(image_path_list), tmin=tmin, tmax=tmax)
        sequence = [read_img(image_path_list[i],dataset_name, 
                csl_cut=transform_cfg.get('csl_cut',True),
                csl_resize=transform_cfg.get('csl_resize',[320,320])) for i in selected_indexs]
        if dataset_name.lower() == 'tvb':
            sequence = [a.resize((256,256)) for a in sequence]
        #transformed_sequence = transform(sequence)
        pseudo_outputs = {'name_sequence':[], 'selected_indexs':[]}
        if is_pseudo:
            id2frame =[]
            for name, (start, end) in zip(name_sequence, boundary_sequence):
                id2frame.extend([[name, ii] for ii in range(start, end+1)])
            selected_frame = [id2frame[ii] for ii in selected_indexs]
            pt = 0 
            for name in name_sequence:
                inds = []
                while pt < len(selected_frame) and selected_frame[pt][0]==name:
                    inds.append(selected_frame[pt][1])
                    pt += 1
                if inds!=[]:
                    pseudo_outputs['name_sequence'].append(name)
                    pseudo_outputs['selected_indexs'].append(inds)
        return sequence, valid_len, selected_indexs, pseudo_outputs
    
    elif 'wlasl' in dataset_name.lower(): #read videos (ISLR) (does not support create pseudo label)
        assert len(name_sequence)==1 and len(boundary_sequence)==1
        num_frames = boundary_sequence[0][1]-boundary_sequence[0][0]+1 #'-2' already done in make_dataset.ipynb
        selected_indexs, valid_len = get_selected_indexs(num_frames, tmin=tmin, tmax=tmax)
        path = datasetname2format[dataset_name.lower()].format(zip_file, name_sequence[0])
        video_byte = ZipReader.read(path)
        inds = sorted(set(selected_indexs))
        i2pos = {i:pos for pos, i in enumerate(inds)}
        video_arrays = _load_frame_nums_to_4darray(video_byte, inds)#T,H,W,3
        video_imgs = [Image.fromarray(ar) for ar in video_arrays]
        sequence = [video_imgs[i2pos[i]] for i in selected_indexs]

        return sequence, valid_len, selected_indexs, {} #no pseudo        

    elif dataset_name.lower() in ['how2sign','cslr']: #read videos (does not support create pseudo label)
        selected_indexs, valid_len = get_selected_indexs(num_frames-2, tmin=tmin, tmax=tmax)
        if dataset_name.lower()=='how2sign':
            path = zip_file+'@realigned_crops/{}.mp4'.format(name)
        elif dataset_name.lower()=='cslr':
            person, s1, s2,_ = name.split('_')
            n1, n2 = int(s1[1:]), int(s2)
            sid = (n1-1)*10+n2
            path = zip_file+'@color-sentence_512x512_2/{:03d}/{}._color.mp4'.format(sid, name)
        else:
            raise ValueError
        video_byte = ZipReader.read(path)
        inds = sorted(set(selected_indexs))
        i2pos = {i:pos for pos, i in enumerate(inds)}
        video_arrays = _load_frame_nums_to_4darray(video_byte, inds)#T,H,W,3
        sequence = [video_arrays[i2pos[i]] for i in selected_indexs]
        if dataset_name.lower()=='cslr':
            h, w = transform_cfg.get('csl_resize',[320,320])
            sequence = [Image.fromarray(a).resize((h,w)) for a in sequence]
        #sequence = transform(sequence)
        return sequence, valid_len, selected_indexs, {}
    else:
        raise ValueError        
