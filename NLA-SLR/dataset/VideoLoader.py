import os, numpy as np
from utils.zipreader import ZipReader
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
    decoded_frames, width, height = lintel.loadvid_frame_nums(video, frame_nums=frame_nums)
    decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
    decoded_frames = np.reshape(
        decoded_frames,
        newshape=(-1, height, width, 3))

    return decoded_frames


def read_img(path):
    zip_data = ZipReader.read(path)
    rgb_im = Image.open(io.BytesIO(zip_data)).convert('RGB')
    return rgb_im


def read_jpg(zip_file, dataset_name, decoded_frames, seq_len, img_dir):
    video_arrays = []
    for f in decoded_frames:
        # assert f<seq_len, (f, seq_len, img_dir)
        if 'MSASL' in dataset_name:
            img_path = '{}@{}{:04d}.png'.format(zip_file, img_dir, f)
        elif 'NMFs-CSL' in dataset_name:
            img_path = '{}@{}image_{:05d}.jpg'.format(zip_file, img_dir, f+1)
        elif dataset_name.lower()=='how2sign':
            img_path = '{}@{}{:04d}.png'.format(zip_file, img_dir, f)
        try:
            img = read_img(img_path)
        except:
            # print('broken img: ', img_path)
            img = np.array(video_arrays[-1])
        video_arrays.append(img) #H,W,C
    video_arrays = np.stack(video_arrays, axis=0) #T,H,W,C
    return video_arrays


def get_selected_indexs(vlen, num_frames=64, is_train=True, setting=['consecutive', 'pad', 'central', 'pad']):
    pad = None  #pad denotes the number of padding frames
    assert len(setting) == 4
    # denote train > 64, test > 64, test < 64
    train_p, train_m, test_p, test_m = setting
    assert train_p in ['consecutive', 'random']
    assert train_m in ['pad']
    assert test_p in ['central', 'start', 'end']
    assert test_m in ['pad', 'start_pad', 'end_pad']
    if num_frames > 0:
        assert num_frames%4 == 0
        if is_train:
            if vlen > num_frames:
                if train_p == 'consecutive':
                    start = np.random.randint(0, vlen - num_frames, 1)[0]
                    selected_index = np.arange(start, start+num_frames)
                elif train_p == 'random':
                    # random sampling
                    selected_index = np.arange(vlen)
                    np.random.shuffle(selected_index)
                    selected_index = selected_index[:num_frames]  #to make the length equal to that of no drop
                    selected_index = sorted(selected_index)
                else:
                    selected_index = np.arange(0, vlen)
            elif vlen < num_frames:
                if train_m == 'pad':
                    remain = num_frames - vlen
                    selected_index = np.arange(0, vlen)
                    pad_left = np.random.randint(0, remain, 1)[0]
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                else:
                    selected_index = np.arange(0, vlen)
            else:
                selected_index = np.arange(0, vlen)
        
        else:
            if vlen >= num_frames:
                start = 0
                if test_p == 'central':
                    start = (vlen - num_frames) // 2
                elif test_p == 'start':
                    start = 0
                elif test_p == 'end':
                    start = vlen - num_frames
                selected_index = np.arange(start, start+num_frames)
            else:
                remain = num_frames - vlen
                selected_index = np.arange(0, vlen)
                if test_m == 'pad':
                    pad_left = remain // 2
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                elif test_m == 'start_pad':
                    pad_left = 0
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                elif test_m == 'end_pad':
                    pad_left = remain
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                else:
                    selected_index = np.arange(0, vlen)
    else:
        # for statistics
        selected_index = np.arange(vlen)

    return selected_index, pad


def pil_list_to_tensor(pil_list, int2float=True):
    func = torchvision.transforms.PILToTensor()
    tensors = [func(pil_img) for pil_img in pil_list]
    #list of C H W
    tensors = torch.stack(tensors, dim=0)
    if int2float:
        tensors = tensors/255
    return tensors #T,C,H,W


def pad_array(array, l_and_r):
    left, right = l_and_r
    if left > 0:
        pad_img = array[0]
        pad = np.tile(np.expand_dims(pad_img, axis=0), tuple([left]+[1]*(len(array.shape)-1)))
        array = np.concatenate([pad, array], axis=0)
    if right > 0:
        pad_img = array[-1]
        pad = np.tile(np.expand_dims(pad_img, axis=0), tuple([right]+[1]*(len(array.shape)-1)))
        array = np.concatenate([array, pad], axis=0)
    return array


def load_video(zip_file, name, vlen, num_frames, dataset_name, is_train, 
                index_setting=['consecutive', 'pad', 'central', 'pad'], temp_scale=[1.0,1.0], ori_vfile=''):
    if 'WLASL' in dataset_name:
        vlen = vlen - 2  # a bug in lintel when load .mp4, by yutong

    selected_index, pad = get_selected_indexs(vlen, num_frames, is_train, index_setting)

    if 'WLASL' in dataset_name:
        video_file = 'WLASL2000/{:s}.mp4'.format(name)
        path = zip_file+'@'+video_file
        video_byte = ZipReader.read(path)
        video_arrays = _load_frame_nums_to_4darray(video_byte, selected_index) #T,H,W,3
    elif 'MSASL' in dataset_name or 'NMFs-CSL' in dataset_name:
        video_arrays = read_jpg(zip_file, dataset_name, selected_index, vlen, ori_vfile)
    
    # pad
    if pad is not None:
        video_arrays = pad_array(video_arrays, pad)
    return video_arrays, selected_index, pad


def load_batch_video(zip_file, names, vlens, dataset_name, is_train, 
                    num_output_frames=64, name2keypoint=None, index_setting=['consecutive','pad','central','pad'], 
                    temp_scale=[1.0,1.0], ori_video_files=[], from64=False):
    #load_video and keypoints, used in collate_fn
    sgn_videos, sgn_keypoints = [], []
    
    batch_videos, batch_keypoints = [], []
    for name, vlen, ori_vfile in zip(names, vlens, ori_video_files):
        video, selected_index, pad = load_video(zip_file, name, vlen, num_output_frames, dataset_name, is_train, index_setting, temp_scale, ori_vfile)
        # video = torch.tensor(video).to(torch.uint8)
        video = torch.tensor(video).float()  #T,H,W,C
        if 'NMFs-CSL' in dataset_name:
            video = torchvision.transforms.functional.resize(video.permute(0,3,1,2), [256,256]).permute(0,2,3,1)
        video /= 255
        batch_videos.append(video) #wo transformed!!
        
        if name2keypoint != None:
            kps = name2keypoint[name][selected_index,:,:]
            if pad is not None:
                kps = pad_array(kps, pad)

            batch_keypoints.append(torch.from_numpy(kps).float()) # T,N,3
        else:
            batch_keypoints.append(None)

    batch_videos = torch.stack(batch_videos, dim=0).permute(0,1,4,2,3) #B,T,C,H,W for spatial augmentation

    if name2keypoint != None:
        batch_keypoints = torch.stack(batch_keypoints, dim=0) #B,T,N,3
    else:
        batch_keypoints = None
    
    sgn_videos.append(batch_videos)
    sgn_keypoints.append(batch_keypoints)
    
    if from64:
        #32-frame counterpart
        if is_train:
            st = np.random.randint(0, num_output_frames//2+1, 1)[0]
        else:
            st = num_output_frames//4
        end = st + num_output_frames//2
        
        sgn_videos.append(sgn_videos[-1][:, st:end, ...])
        if sgn_keypoints[-1] is None:
            sgn_keypoints.append(None)
        else:
            sgn_keypoints.append(sgn_keypoints[-1][:, st:end, ...])
    
    return sgn_videos, sgn_keypoints

