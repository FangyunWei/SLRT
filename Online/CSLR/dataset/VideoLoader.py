import os, numpy as np
from utils.zipreader import ZipReader
import io, torch, torchvision
from PIL import Image
import lintel, random


def compute_iou(gt_start, gt_end, vlen, center, win_size=16):
    #gt_end should be inclusive
    left = center - (win_size//2)
    right = left + win_size - 1
    left = max(0, left)
    right = min(vlen-1, right)

    a = min(right, gt_end)
    b = max(left, gt_start)

    interarea = a - b + 1
    interarea = max(interarea, 0)
    win_area = right - left + 1
    gt_area = gt_end - gt_start + 1
    return interarea/(win_area+gt_area-interarea)


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


def read_img(path, dataset_name, csl_cut=False, csl_resize=-1):
    zip_data = ZipReader.read(path)
    rgb_im = Image.open(io.BytesIO(zip_data)).convert('RGB')
    if dataset_name.lower() in ['csl', 'csl_iso', 'cslr']: #cslr won'r enter here
        if csl_cut:
            rgb_im = rgb_im.crop((0,80,512,512))
        if csl_resize!=-1:
            if csl_cut:
                assert csl_resize==[320,270] #width height
            else:
                assert csl_resize[0]==csl_resize[1]
            rgb_im = rgb_im.resize((csl_resize[0], csl_resize[1]))
    return rgb_im


def read_jpg(zip_file, dataset_name, decoded_frames, seq_len, img_dir):
    video_arrays = []
    for f in decoded_frames:
        # assert f<seq_len, (f, seq_len, img_dir)
        if dataset_name.lower() in ['csl', 'csl_iso']:
            img_path = '{}@sentence_frames-512x512/{}/{:06d}.jpg'.format(zip_file, img_dir, f)
        elif dataset_name.lower() in ['phoenix_iso', 'phoenix']:
            img_path = '{}@images/{}/images{:04d}.png'.format(zip_file, img_dir, f+1) #start from 1
        elif dataset_name.lower() in ['phoenix2014_iso', 'phoenix2014']:
            img_path = '{}@{}.avi_pid0_fn{:06d}-0.png'.format(zip_file, img_dir, f)
        elif 'MSASL' in dataset_name:
            img_path = '{}@{}{:04d}.png'.format(zip_file, img_dir, f)
        elif 'NMFs-CSL' in dataset_name:
            img_path = '{}@{}image_{:05d}.jpg'.format(zip_file, img_dir, f+1)
        elif dataset_name.lower()=='how2sign':
            img_path = '{}@{}{:04d}.png'.format(zip_file, img_dir, f)
        try:
            img = read_img(img_path, dataset_name, csl_cut=False, csl_resize=[320,320])
        except:
            # print('broken img: ', img_path)
            img = np.array(video_arrays[-1])
        #sentence_frames-512x512.zip@sentence_frames-512x512/S000853_P0000_T00/000000.jpg
        #PHOENIX2014T_videos.zip@images/train/03June_2011_Friday_tagesschau-7638/images0014.png
        video_arrays.append(img) #H,W,C
    video_arrays = np.stack(video_arrays, axis=0) #T,H,W,C
    return video_arrays


def get_selected_indexs(vlen, num_frames=64, is_train=True, setting=['consecutive', 'pad', 'central', 'pad']):
    pad = None  #pad denotes the number of padding frames
    assert len(setting) == 4
    # denote train > 64, test > 64, test < 64
    train_p, train_m, test_p, test_m = setting
    assert train_p in ['consecutive', 'random', 'temp_scale', 'crop_then_pad']
    assert train_m in ['pad', 'even', 'temp_scale']
    assert test_p in ['central', 'start', 'end', 'left_mid', 'right_mid']
    assert test_m in ['pad', 'start_pad', 'end_pad', 'left_mid_pad', 'right_mid_pad', 'even']
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
                if train_p == 'crop_then_pad' and random.random() > 0.5:
                    start = np.random.randint(0, num_frames//2+1, 1)[0]
                    selected_index = np.arange(start, start+num_frames//2)
                    remain = num_frames - num_frames//2
                    pad_left = np.random.randint(0, remain, 1)[0]
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                else:
                    selected_index = np.arange(0, vlen)
        
        else:
            if vlen >= num_frames:
                start = 0
                if test_p == 'central':
                    start = (vlen - num_frames) // 2
                elif test_p == 'start':
                    start = 0
                elif test_p == 'left_mid':
                    start = (vlen - num_frames) // 4
                elif test_p == 'right_mid':
                    start = int((vlen - num_frames) / 4 * 3)
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
                elif test_m == 'left_mid_pad':
                    pad_left = remain // 4
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                elif test_m == 'right_mid_pad':
                    pad_left = int(remain / 4 * 3)
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


# def read_img(path, dataset_name, csl_cut, csl_resize=-1):
#     zip_data = ZipReader.read(path)
#     rgb_im = Image.open(io.BytesIO(zip_data)).convert('RGB')    
#     if dataset_name.lower()=='csl':
#         if csl_cut:
#             rgb_im = rgb_im.crop((0,80,512,512))
#         if csl_resize!=-1:
#             if csl_cut:
#                 assert csl_resize==[320,270] #width height
#             else:
#                 assert csl_resize[0]==csl_resize[1]
#             rgb_im = rgb_im.resize((csl_resize[0], csl_resize[1]))
#     return rgb_im


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


def even_replicate(vlen, num_frames=64, pre_index=None, is_train=True):
    if pre_index is None:
        pre_index = [i for i in range(vlen)]
    num_remain_frames = num_frames - vlen
    copied_index = []
    if num_remain_frames <= vlen:
        intv = vlen // num_remain_frames
        for i in range(num_remain_frames):
            if is_train:
                idx = random.sample(pre_index[i*intv:(i+1)*intv], 1)[0]
            else:
                idx = (i*intv+(i+1)*intv) // 2
            copied_index.append(idx)
    else:
        ratio = num_remain_frames / vlen
        integer = int(ratio)
        remainder = num_remain_frames - integer*vlen
        copied_index.extend(pre_index*integer)
        if remainder != 0:
            intv = vlen // remainder
            for i in range(remainder):
                if is_train:
                    # try:
                    idx = random.sample(pre_index[i*intv:(i+1)*intv], 1)[0]
                    # except:
                    #     print(i, intv, len(pre_index))
                else:
                    idx = (i*intv+(i+1)*intv) // 2
                copied_index.append(idx)
    copied_index = np.array(copied_index)
    selected_index = sorted(np.concatenate([np.array(pre_index), copied_index]))
    return selected_index


def load_video(zip_file, name, vlen, raw_vlen, num_frames, dataset_name, is_train, 
                index_setting=['consecutive', 'pad', 'central', 'pad'], temp_scale=[1.0,1.0], ori_vfile=''):
    if 'WLASL' in dataset_name or 'SLR500' in dataset_name:
        vlen = vlen - 2  # a bug in lintel when load .mp4

    selected_index, pad = get_selected_indexs(vlen, num_frames, is_train, index_setting)

    if dataset_name in ['phoenix_iso', 'phoenix2014_iso', 'phoenix_comb_iso', 'csl_iso']:
        s = int(name[name.find('[')+1:name.find(':')])
        e = int(name[name.find(':')+1:name.find(']')])
        # if is_train:
        #     winsize = num_frames
        #     if 'blank' in name:
        #         iou_thr = 0.5
        #     else:
        #         iou_thr = 0.3
        #     iou = np.zeros(raw_vlen)
        #     for cen in range(0, raw_vlen, 1):
        #         iou[cen] = compute_iou(s, e-1, raw_vlen, cen, winsize)
        #     iou_mask = (iou>iou_thr)
        #     if np.sum(iou_mask) >= 1:
        #         idx = np.where(iou_mask==1)[0]
        #         cen = random.sample(list(idx), 1)[0]
        #         s = cen - (winsize//2)
        #         e = s + winsize - 1
        #         s = max(0, s)
        #         e = min(raw_vlen-1, e) + 1
        #         vlen = e - s
        #         selected_index, pad = get_selected_indexs(vlen, num_frames, is_train, index_setting)

        # pad from raw video
        # if pad is not None:
        #     pad_left, pad_right = pad
        #     temp_s = s - pad_left
        #     temp_e = e - pad_right
        selected_index = np.arange(s,e)[selected_index]
        # print(name, s, e, selected_index)


    if 'WLASL' in dataset_name or 'SLR500' in dataset_name:
        if 'WLASL' in dataset_name:
            if 'crop' in zip_file:
                video_file = 'WLASL2000_crop/{:s}.mp4'.format(name)
            else:
                video_file = 'WLASL2000/{:s}.mp4'.format(name)
        elif 'SLR500' in dataset_name:
            video_file = ori_vfile
        path = zip_file+'@'+video_file
        video_byte = ZipReader.read(path)
        video_arrays = _load_frame_nums_to_4darray(video_byte, selected_index) #T,H,W,3
    elif 'MSASL' in dataset_name or 'NMFs-CSL' in dataset_name or \
        dataset_name in ['phoenix_iso', 'phoenix2014_iso', 'phoenix_comb_iso', 'phoenix', 'phoenix2014', 'phoenixcomb', 'csl', 'csl_iso']:
        if dataset_name in ['phoenix_comb_iso', 'phoenixcomb']:
            if 'fullFrame' in ori_vfile:
                real_datasetname = 'phoenix2014_iso' if 'iso' in dataset_name else 'phoenix2014'
                video_arrays = read_jpg(zip_file, real_datasetname, selected_index, vlen, ori_vfile)
            else:
                real_datasetname = 'phoenix_iso' if 'iso' in dataset_name else 'phoenix'
                video_arrays = read_jpg(zip_file, real_datasetname, selected_index, vlen, ori_vfile)
        else:
            video_arrays = read_jpg(zip_file, dataset_name, selected_index, vlen, ori_vfile)

    train_p, train_m, test_p, test_m = index_setting
    if is_train:
        if train_p == 'temp_scale':
            tmin, tmax = temp_scale
            min_len = int(tmin*vlen)
            max_len = int(tmax*vlen)
            if max_len < min_len:
                max_len = min_len
            selected_len = np.random.randint(min_len, max_len+1)
            if selected_len <= vlen:
                selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
            else:
                copied_index = np.random.randint(0, vlen, selected_len-vlen)
                selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))

            # sample to 64 frames
            vlen = len(selected_index)
            if len(selected_index) > num_frames:
                # consecutive sampling
                start = np.random.randint(0, vlen - num_frames, 1)[0]
                selected_index = selected_index[start:start+num_frames]
            elif len(selected_index) < num_frames:
                # even replicate
                selected_index = even_replicate(vlen, num_frames, pre_index=selected_index, is_train=is_train)
            # print(vlen)
            video_arrays = video_arrays[selected_index]

        elif vlen < num_frames and train_m == 'even':
            selected_index = even_replicate(vlen, num_frames, is_train=is_train)
            video_arrays = video_arrays[selected_index]
    
    else:
        if vlen < num_frames and test_m == 'even':
            selected_index = even_replicate(vlen, num_frames, is_train=is_train)
            video_arrays = video_arrays[selected_index]
    
    # pad
    if pad is not None:
        video_arrays = pad_array(video_arrays, pad)
    return video_arrays, selected_index, pad
    # else:
    #     raise ValueError


def load_batch_video(zip_file, names, vlens, raw_vlens, dataset_name, is_train, 
                    num_output_frames=64, name2keypoint=None, index_setting=['consecutive','pad','central','pad'], temp_scale=[1.0,1.0],
                    ori_video_files=[], fps=1, from64=False):
    #load_video and keypoints, used in collate_fn
    sgn_videos, sgn_keypoints = [], []
    if type(num_output_frames) == int:
        num_output_frames = [num_output_frames]
    if type(fps) == int:
        fps = [fps]*len(num_output_frames)
    
    for n_frames, f in zip(num_output_frames, fps):
        batch_videos, batch_keypoints = [], []
        for name, vlen, raw_vlen, ori_vfile in zip(names, vlens, raw_vlens, ori_video_files):
            video, selected_index, pad = load_video(zip_file, name, vlen, raw_vlen, n_frames, dataset_name, is_train, index_setting, temp_scale, ori_vfile)
            # video = torch.tensor(video).to(torch.uint8)
            video = torch.tensor(video).float()  #T,H,W,C
            if 'NMFs-CSL' in dataset_name:
                video = torchvision.transforms.functional.resize(video.permute(0,3,1,2), [256,256]).permute(0,2,3,1)
            video /= 255
            batch_videos.append(video) #wo transformed!!
            
            if name2keypoint != None:
                if dataset_name in ['phoenix_iso', 'phoenix2014_iso', 'phoenix_comb_iso', 'csl_iso']:
                    kps = name2keypoint[ori_vfile][selected_index,:,:]
                else:
                    kps = name2keypoint[name][selected_index,:,:]
                if pad is not None:
                    kps = pad_array(kps, pad)

                # ------------------------------clean pose-----------------------------------
                # kps_of_interest = kps[:, (0,21), 1]  #[T,2], height of two hands
                # kps_of_interest /= 256
                # hand_left, hand_right = kps_of_interest[:, 0], kps_of_interest[:, 1]
                # thr = 0.9
                # mask = np.logical_or(hand_left<thr, hand_right<thr)  #T
                # mask = np.where(mask > 0)[0]
                # try:
                #     start, end = min(mask), max(mask)+1
                # except:
                #     start, end = 0, num_output_frames
                # pad = (start, num_output_frames-end)
                # sgn_videos[-1] = torch.tensor(pad_array(sgn_videos[-1].numpy()[start:end], pad)).float()
                # kps = pad_array(kps[start:end], pad)
                # ------------------------------clean pose-----------------------------------

                batch_keypoints.append(torch.from_numpy(kps).float()) # T,N,3
            else:
                batch_keypoints.append(None)
    
        batch_videos = torch.stack(batch_videos, dim=0).permute(0,1,4,2,3) #B,T,C,H,W for spatial augmentation
        if f==2:
            batch_videos = batch_videos[:, ::2, ...]

        if name2keypoint != None:
            batch_keypoints = torch.stack(batch_keypoints, dim=0) #B,T,N,3
            if f==2:
                batch_keypoints = batch_keypoints[:, ::2, ...]
        else:
            batch_keypoints = None
        
        sgn_videos.append(batch_videos)
        sgn_keypoints.append(batch_keypoints)
    
    st = None
    if from64:
        if from64 == 'slowfast':
            st = 0
            end = num_output_frames[0]
            stride = 2
        else:
            if (is_train and from64!='always_central') or from64=='random':
                st = np.random.randint(0, num_output_frames[0]//2+1, 1)[0]
            else:
                st = num_output_frames[0]//4
            end = st + num_output_frames[0]//2
            stride = 1
        
        sgn_videos.append(sgn_videos[-1][:, st:end:stride, ...])
        if sgn_keypoints[-1] is None:
            sgn_keypoints.append(None)
        else:
            sgn_keypoints.append(sgn_keypoints[-1][:, st:end:stride, ...])
    
    return sgn_videos, sgn_keypoints, st


# def collate_batch(sgn_videos, sgn_keypoints, frame_num=64):
#     if type(frame_num) == list:
#         real_frame_num = random.sample(frame_num, 1)[0]
#     else:
#         real_frame_num = frame_num

#     for 
