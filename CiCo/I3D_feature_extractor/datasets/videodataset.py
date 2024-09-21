import math
import os
import pickle as pkl
import random
from abc import abstractmethod
from collections import OrderedDict, defaultdict

import cv2
import numpy as np
import torch
import torch.utils.data as data
from beartype import beartype

from utils.imutils import (im_to_numpy, im_to_torch, im_to_video,
                           resize_generic, video_to_im)
from utils.transforms import (bbox_format, color_normalize, im_color_jitter,
                              scale_yxyx_bbox)

cv2.setNumThreads(0)

import copy
# not sure how many video readers it is safe to keep open for a given video
# but this seems reasonable. For safety, we sho
CAP_CACHE_LIMIT = 10
DISABLE_CACHING = True


class VideoDataset(data.Dataset):
    def __init__(self):
        self.mean = 0.5 * torch.ones(3)
        self.std = 1.0 * torch.ones(3)
        if not hasattr(self, "use_bbox"):
            self.use_bbox = False
        self._set_datasetname()
        print(f"VideoDataset {self.datasetname.upper()} {self.setname} ({len(self)})")

    @abstractmethod
    def _set_datasetname(self):
        raise NotImplementedError(f"Dataset name must be implemented by subclasss")

    @abstractmethod
    def _get_nframes(self, ind):
        raise NotImplementedError(f"_get_nframes name must be implemented by subclasss")

    def _slide_windows(self, valid,stride):
        test = []
        t_beg = []
        num_clip_list=[]
        # For each video
        for i, k in enumerate(valid):
            init_t, end_t, len_t, nFrames = self._get_valid_temporal_interval(k)
            assert nFrames == self._get_nframes(k)

            num_clips = math.ceil((len_t - self.num_in_frames) / stride) + 1
            if num_clips<=0:
                num_clips=1
            num_clip_list.append(num_clips)
            # For each clip
            for j in range(num_clips):
                # Check if there are enough frames for num_in_frames.
                actual_clip_length = min(self.num_in_frames, len_t - j * stride)
                if actual_clip_length == self.num_in_frames:
                    t_beg.append(init_t + j * stride)
                else:
                    # If not enough frames, reduce the stride for the last clip
                    if end_t - self.num_in_frames>=0:
                        t_beg.append(end_t - self.num_in_frames)
                    else:
                        t_beg.append(0)
                test.append(k)

        t_beg = np.asarray(t_beg)
        valid = np.asarray(test)
        return valid, t_beg,num_clip_list

    def _load_rgb(self, ind, frame_ix):
        """Loads the video frames from file
            frame_ix could be range(t, t + nframes) for consecutive reading
                or a random sorted subset of [0, video_length] of size nframes
        """

        is_consecutive = range(min(frame_ix), max(frame_ix) + 1) == frame_ix
        nframes = len(frame_ix)
        if nframes<16:
            nframes=16
            frame_ix=range(16)
        videofile =ind
        use_cv2 = True
        if getattr(self, "video_data_dict", False):
            use_cv2 = False
            compressed_frames = self.video_data_dict[videofile]["data"]
        elif getattr(self, "featurize_mode", False) and not DISABLE_CACHING:
            cap = None
            if not hasattr(self, "cached_caps"):
                self.cached_caps = defaultdict(OrderedDict)
            if videofile in self.cached_caps:
                cap = self.cached_caps[videofile].pop(frame_ix.start, None)
                assert is_consecutive, "capture caching should only use consecutive ims"
            if not cap:
                cap = cv2.VideoCapture(videofile)
                # Do the frame setting only once if the rest are consecutive
                if is_consecutive:
                    cap.set(propId=1, value=frame_ix[0])
        else:
            cap = cv2.VideoCapture(videofile)
            # Do the frame setting only once if the rest are consecutive
            if is_consecutive:
                cap.set(propId=1, value=frame_ix[0])

        # Frame reads
        if getattr(self, "gpu_collation", False):
            msg = "expected collation dim == 256"
            assert self.gpu_collation == 256, msg
            rgb = torch.zeros(3, nframes, self.gpu_collation, self.gpu_collation)
        else:
            rgb = torch.zeros(
                3, nframes, self._get_img_height(ind), self._get_img_width(ind)
            )
        # rgb = torch.zeros(3, nframes, self.img_height, self.img_width)
        for f, fix in enumerate(frame_ix):
            if use_cv2:
                if not is_consecutive:
                    cap.set(propId=1, value=fix)
                # frame: BGR, (240, 320, 3), dtype=uint8 0..255
                ret, frame = cap.read()
            else:
                ret = fix < len(compressed_frames)
                if ret:
                    frame = Image.open(compressed_frames[fix])

            if ret:
                if use_cv2:
                    # BGR (OpenCV) to RGB (Torch)
                    frame = frame[:, :, [2, 1, 0]]
                # CxHxW (3, 240, 320), 0..1 --> np.transpose(frame, [2, 0, 1]) / 255.0
                rgb_t = im_to_torch(frame)
                rgb[:, f, :, :] = rgb_t
            else:
                # Copy last frame for temporal padding
                rgb[:, f, :, :] = rgb[:, f - 1, :, :]
        if use_cv2:
            if (
                hasattr(self, "featurize_mode")
                and self.featurize_mode
                and not DISABLE_CACHING
            ):
                if fix == self._get_nframes(ind):
                    cap.release()
                else:
                    # store pointer to avoid duplicate decoding
                    self.cached_caps[videofile][frame_ix.stop] = cap
                    if len(self.cached_caps[videofile]) > CAP_CACHE_LIMIT:
                        # we rely on OrderedDict to preserve key order
                        oldest_keys = list(self.cached_caps.keys())
                        print(
                            f"Cache overflow [{len(self.cached_caps[videofile])}]"
                            f" >{CAP_CACHE_LIMIT}, clearing half of the keys"
                        )
                        for old_key in oldest_keys[: CAP_CACHE_LIMIT // 2]:
                            # To guard against race conditions we supply a default for
                            # missing keys
                            self.cached_caps[videofile].pop(old_key, None)
            else:
                cap.release()

        # rgb.view(3 * nframes, rgb.size(2), rgb.size(3))
        rgb = video_to_im(rgb)
        return rgb

    def collate_fn(self, batch):
        """Note: To enable usage with ConcatDataset, this must not rely on any attributes
        that are specific to a current dataset (apart from `gpu_collation`), since a
        single collate_fn will be shared by all datasets.
        """
        if not getattr(self, "gpu_collation", False):
            return torch.utils.data._utils.collate.default_collate(batch)

        meta = { "index", "data_index", "dataset",'frame'}
        minibatch = {key: [x[key] for x in batch] for key in meta}
        rgb = torch.stack([x["rgb"] for x in batch])
        for key_long_dtype in {"index"}:
            minibatch[key_long_dtype] = torch.LongTensor(minibatch[key_long_dtype])

        minibatch["rgb"] = rgb
        # TODO(Samuel): could probably simplify the treatment of `class_names` - should
        # clean this up.  For now, we mimic the existing data structure, which duplicates
        # the class name once for each batch item.

        minibatch["gpu_collater"] = True
        return minibatch

    def gpu_collater(self, minibatch, concat_datasets=None):
        rgb = minibatch["rgb"]
        assert rgb.is_cuda, "expected tensor to be on the GPU"
        if self.setname == "train":
            is_hflip = random.random() < self.hflip
            if is_hflip:
                # horizontal axis is last
                rgb = torch.flip(rgb, dims=[-1])

        if self.setname == "train":
            rgb = im_color_jitter(rgb, num_in_frames=self.num_in_frames, thr=0.2)

        # For now, mimic the original pipeline.  If it's still a bottleneck, we should
        # collapse the cropping, resizing etc. logic into a single sampling grid.
        iB, iC, iK, iH, iW = rgb.shape
        assert iK == self.num_in_frames, "unexpected number of frames per clip"

        bbox_yxyx = np.zeros((iB, 4), dtype=np.float32)
        for ii, data_index in enumerate(minibatch["data_index"]):
            bbox_yxyx[ii] = np.array([0, 0, 1, 1])
            # Otherwise, it fails when mixing use_bbox True and False for two datasets
            if concat_datasets is not None:
                local_use_bbox = concat_datasets[minibatch["dataset"][ii]].use_bbox
            else:
                local_use_bbox = self.use_bbox
            if local_use_bbox:
                # Until we patch ConcatDataset, we need to pass the dataset object
                # explicitly to handle bbox selection
                if concat_datasets is not None:
                    get_bbox = concat_datasets[minibatch["dataset"][ii]]._get_bbox
                else:
                    get_bbox = self._get_bbox
                bbox_yxyx[ii] = get_bbox(data_index)

        # require that the original boxes lie inside the image
        bbox_yxyx[:, :2] = np.maximum(0, bbox_yxyx[:, :2])
        bbox_yxyx[:, 2:] = np.minimum(1, bbox_yxyx[:, 2:])

        if self.setname == "train":
            if is_hflip:
                flipped_xmin = 1 - bbox_yxyx[:, 3]
                flipped_xmax = 1 - bbox_yxyx[:, 1]
                bbox_yxyx[:, 1] = flipped_xmin
                bbox_yxyx[:, 3] = flipped_xmax

            # apply a random (isotropic) scale factor to box coordinates
            rand_scale = np.random.rand(iB, 1)
            rand_scale = 1 - self.scale_factor + 2 * self.scale_factor * rand_scale
            # Mimic the meaning of scale used in CPU pipeline
            rand_scale = 1 / rand_scale
            bbox_yxyx = scale_yxyx_bbox(bbox_yxyx, scale=rand_scale)

        # apply random/center cropping to match the proportions used in the original code
        # (the scaling is not quite identical, but close to it)
        if self.setname == "train":
            crop_box_sc = (self.inp_res / self.resize_res) * rand_scale
        else:
            crop_box_sc = self.inp_res / self.resize_res
        bbox_yxyx = scale_yxyx_bbox(bbox_yxyx, scale=crop_box_sc)

        # If training, jitter the location such that it still lies within the appropriate
        # region defined by the (optionally scaled) bounding box
        if self.setname == "train":
            crop_bbox_cenhw = bbox_format(bbox_yxyx, src="yxyx", dest="cenhw")
            cropped_hw = crop_bbox_cenhw[:, 2:]
            valid_offset_region_hw = ((1 - crop_box_sc) / crop_box_sc) * cropped_hw
            valid_offset_samples = np.random.rand(iB, 2)
            valid_rand_offsets = (valid_offset_samples - 0.5) * valid_offset_region_hw
            # apply offsets
            bbox_yxyx += np.tile(valid_rand_offsets, (1, 2))

        # TODO(Samuel): go back over:
        #  (1) the corner alignment logic to check we are doing # the right thing here.
        #  (2) whether zero padding is appropriate for out-of-bounds handling
        # center in [-1, -1] coordinates
        bbox_yxyx = 2 * bbox_yxyx - 1
        grids = torch.zeros(
            iB, self.inp_res, self.inp_res, 2, device=rgb.device, dtype=rgb.dtype
        )

        for ii, bbox in enumerate(bbox_yxyx):
            yticks = torch.linspace(start=bbox[0], end=bbox[2], steps=self.inp_res)
            xticks = torch.linspace(start=bbox[1], end=bbox[3], steps=self.inp_res)
            grid_y, grid_x = torch.meshgrid(yticks, xticks)
            # The grid expects the ordering to be x then y
            grids[ii] = torch.stack((grid_x, grid_y), 2)

        # merge RGB and clip dimensions to use with grid sampler
        rgb = rgb.view(rgb.shape[0], 3 * self.num_in_frames, iH, iW)
        rgb = torch.nn.functional.grid_sample(
            rgb, grid=grids, mode="bilinear", align_corners=False, padding_mode="zeros",
        )
        # unflatten channel/clip dimension
        rgb = rgb.view(rgb.shape[0], 3, self.num_in_frames, self.inp_res, self.inp_res)
        rgb = color_normalize(rgb, self.mean, self.std)
        minibatch["rgb"] = rgb
        return minibatch

    def _get_single_video(self, index, data_index, frame_ix):
        """Loads/augments/returns the video data
        :param index: Index wrt to the data loader
        :param data_index: Index wrt to train/valid list
        :param frame_ix: A list of frame indices to sample from the video
        :return data: Dictionary of input/output and other metadata
        """
        # If the input is pose (Pose->Sign experiments)
        if hasattr(self, "input_type") and self.input_type == "pose":
            data = {
                "rgb": self._get_pose(data_index, frame_ix),
                "index": index,
                "data_index": data_index,
                "dataset": self.datasetname,
            }
            return data
        # Otherwise the input is RGB
        else:
            rgb = self._load_rgb(data_index, frame_ix)
            if getattr(self, "mask_rgb", False):
                rgb = self._mask_rgb(
                    rgb,
                    data_index,
                    frame_ix,
                    region=self.mask_rgb,
                    mask_type=self.mask_type,
                )

        if getattr(self, "gpu_collation", False):
            # Meta info
            data = {
                "rgb": rgb,
                "index": index,
                "data_index": data_index,
                "dataset": self.datasetname,
            }
            return data

        # Preparing RGB data
        if self.setname == "train":
            # Horizontal flip: disable for now, should be done after the bbox cropping
            is_hflip = random.random() < self.hflip
            if is_hflip:
                rgb = torch.flip(rgb, dims=[2])
            # Color jitter
            rgb = im_color_jitter(rgb, num_in_frames=self.num_in_frames, thr=0.2)

        rgb = im_to_numpy(rgb)
        iH, iW, iC = rgb.shape

        if self.use_bbox:
            y0, x0, y1, x1 = self._get_bbox(data_index)
            y0 = max(0, int(y0 * iH))
            y1 = min(iH, int(y1 * iH))
            x0 = max(0, int(x0 * iW))
            x1 = min(iW, int(x1 * iW))
            if self.setname == "train" and is_hflip:
                x0 = iW - x0
                x1 = iW - x1
                x0, x1 = x1, x0
            rgb = rgb[y0:y1, x0:x1, :]
            rgb = resize_generic(
                rgb, self.resize_res, self.resize_res, interp="bilinear", is_flow=False,
            )
            iH, iW, iC = rgb.shape

        resol = self.resize_res  # 300 for 256, 130 for 112 etc.
        if self.setname == "train":
            # Augment the scaled resolution between:
            #     [1 - self.scale_factor, 1 + self.scale_factor)
            rand_scale = random.random()
            resol *= 1 - self.scale_factor + 2 * self.scale_factor * rand_scale
            resol = int(resol)
        if iW > iH:
            nH, nW = resol, int(resol * iW / iH)
        else:
            nH, nW = int(resol * iH / iW), resol
        # Resize to nH, nW resolution
        rgb = resize_generic(rgb, nH, nW, interp="bilinear", is_flow=False)

        # Crop
        if self.setname == "train":
            # Random crop coords
            ulx = random.randint(0, nW - self.inp_res)
            uly = random.randint(0, nH - self.inp_res)
        else:
            # Center crop coords
            ulx = int((nW - self.inp_res) / 2)
            uly = int((nH - self.inp_res) / 2)
        # Crop 256x256
        rgb = rgb[uly : uly + self.inp_res, ulx : ulx + self.inp_res]
        rgb = im_to_torch(rgb)
        rgb = im_to_video(rgb)
        rgb = color_normalize(rgb, self.mean, self.std)

        # Return
        data = {
            "rgb": rgb,
            "class": self._get_class(data_index, frame_ix),
            "index": index,
            "class_names": self.class_names,
            "dataset": self.datasetname,
        }

        return data

    def __getitem__(self, index):
        if self.setname == "train":
            data_index = self.train[index]
        else:
            data_index = self.valid[index]

        frame_ix = self._sample_frames(index, data_index)
        # print(data_index,frame_ix)
        video={}
        video['rgb']=self.videos[data_index]['rgb'][:,frame_ix,:,:]
        video['frame']=frame_ix
        for keys in self.videos[data_index]:
            if keys!='rgb':
                video[keys]= self.videos[data_index][keys]
        video['index']=index
        return video

    def __len__(self):
        if self.setname == "train":
            return len(self.train)
        else:
            return len(self.valid)

    # TODO: Get rid of this
    def set_video_metadata(self, data, meta_key, fixed_sz_frames=None):
        """Set the appropriate metadata for the videos used by the current dataset.

        Args:
            data (dict): a collection of meta data associated with the current dataset.
            meta_key (str): the key under which the video meta data is stored in Gul's
                info data dicts.
            fixed_sz_frames (int): A value to be used for the height and width of frames,
                which, if provided, will overwrite the meta data from the data dict. This
                is used for gpu collation.
        """
        # restrict to the given videos
        if (
            hasattr(self, "featurize_mode")
            and hasattr(self, "featurize_mask")
            and self.featurize_mode
            and self.featurize_mask
        ):
            keep = np.array([self.featurize_mask in x for x in data["videos"]["name"]])
            print(f"Filtered featurization to {keep.sum()} videos")
            assert keep.sum(), f"After filtering, there were no videos to process!"
            for key in {"name", "word", "word_id", "split"}:
                data["videos"][key] = np.array(data["videos"][key])[keep].tolist()
            for subkey, subval in data["videos"]["videos"].items():
                data["videos"]["videos"][subkey] = np.array(subval)[keep].tolist()

        # Each video has a different resolution
        # self.bboxes = [s for s in np.asarray(data['videos']['box'])]
        self.img_widths = data["videos"][meta_key]["W"]
        self.img_heights = data["videos"][meta_key]["H"]
        self.num_frames = data["videos"][meta_key]["T"]

        # Used for GPU collation
        if fixed_sz_frames:
            self.img_widths = [fixed_sz_frames for _ in self.img_widths]
            self.img_heights = [fixed_sz_frames for _ in self.img_heights]

    def set_class_names(self, data, word_data_pkl):
        """Assign the dataset class names, optionally filtering them with a pickle
        file of words.
        """
        if word_data_pkl is None:
            # All words
            subset_ix = range(len(data["videos"]["name"]))
            self.classes = data["videos"]["word_id"]
            with open(os.path.join(self.root_path, "info", "words.txt"), "r") as f:
                self.class_names = f.read().splitlines()
        else:
            print(f"Using the words from {word_data_pkl}")
            word_data = pkl.load(open(word_data_pkl, "rb"))
            self.classes = []
            subset_ix = []
            for i, w in enumerate(data["videos"]["word"]):
                if w in word_data["words"]:
                    self.classes.append(word_data["words_to_id"][w])
                    subset_ix.append(i)
                else:
                    self.classes.append(-1)
            with open(word_data_pkl.replace(".pkl", ".txt"), "r") as f:
                self.class_names = f.read().splitlines()
        return subset_ix

    @beartype
    def get_set_classes(self, is_train: bool = False):
        if is_train:
            return np.asarray(self.classes)[self.train]
        else:
            return np.asarray(self.classes)[self.valid]

    def get_all_classes(self):
        return self.classes

    def get_all_videonames(self):
        return self.videos

    def _get_valid_temporal_interval(self, data_index):
        """Returns the [beginning,end) frame indices
        from which we will sample inputs from the full video.
        By default it returns the full video.
        For bsl1k, num_last_frames=20, it returns the last 20 frames.
        """
        nFrames = self._get_nframes(data_index)
        # By default the full video is used.
        t0 = 0
        t1 = nFrames
        # Only take last num_last_frames of the video (used only for BSL1K)
        if hasattr(self, "num_last_frames"):
            if nFrames >= self.num_last_frames:
                t0 = nFrames - self.num_last_frames
        len_t = t1 - t0
        return t0, t1, len_t, nFrames

    def _sample_frames(self, index, data_index):
        """Returns a list of frame indices to sample (consecutively):
            if train: random initial point
            if val: middle clip
            if evaluate_video: sliding window
        """
        # Temporal interval selection
        init_t, end_t, len_t, nFrames = self._get_valid_temporal_interval(data_index)

        # Consecutively sampled frames
        if self.evaluate_video:
            # TEST (pre-computed frame number for sliding window)
            t = self.t_beg[index]
        else:
            # TRAIN (random initial frame)
            if self.setname == "train":
                max_offset = init_t + max(0, len_t - self.num_in_frames)
                t = random.randint(init_t, max_offset)
            # VAL (middle clip)
            else:
                t = init_t + max(0, math.floor((len_t - self.num_in_frames) / 2))
        frame_ix = range(t, t + self.num_in_frames)
        return frame_ix
