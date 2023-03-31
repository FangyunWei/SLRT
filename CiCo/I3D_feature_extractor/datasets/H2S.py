import math
import os
import pickle as pkl

import cv2
import numpy as np

from datasets.videodataset import VideoDataset

cv2.setNumThreads(0)

from utils.imutils import im_to_video
class H2S(VideoDataset):
    def __init__(
        self,
        root_path="",
        inp_res=224,
        resize_res=256,
        setname="val",
        scale_factor=0.1,
        num_in_frames=16,
        evaluate_video=True,
        hflip=0.5,
        stride=1,
        gpu_collation=False,
        assign_labels="auto",
        rank=0,
        split="val",
        split_size=16,
    ):
        self.root_path = root_path
        self.setname = setname  # train, val or test
        self.gpu_collation = gpu_collation
        self.inp_res = inp_res
        self.resize_res = resize_res
        self.scale_factor = scale_factor
        self.num_in_frames = num_in_frames
        self.evaluate_video = evaluate_video
        self.hflip = hflip
        self.stride = stride
        self.assign_labels = assign_labels
        import pickle
        self.num_frames= {}
        self.rank=rank
        self.video_folder = "videos"
        meta_key = self.video_folder
        self.split=split
        self.datasetname='H2S'
        self.train_file = sorted(os.listdir(os.path.join(self.root_path,self.split)))

        ##split the dataset into multiple pieces
        if self.split=='train':
            N=256
        else:
            N=16
        rank=rank
        print(rank)
        print(f'all videos #{len(self.train_file)}')
        n_per_split = len(self.train_file) // N
        print(f'split to {N} subsets (#~{n_per_split})')
        start, end = rank * n_per_split, (rank + 1) * n_per_split
        if rank == N - 1:
            end = max(len(self.train_file), end)
        print(f'rank {rank}, {start}~{end}')
        self.train_file = self.train_file[start:end]


        self.train=[os.path.join(self.root_path,self.split,v) for v in self.train_file]
        # self.valid = os.listdir(os.path.join(self.root_path,'test'))
        # self.valid=[os.path.join(self.root_path,'test',v) for v in self.valid]


        frame_info_file=f'{self.split}_frame_dict.pkl'
        with open('%s' % (frame_info_file), 'rb') as f:
            frame_info = pickle.load(f)
        self.num_frames = {}
        for train in self.train:
            key = train.split('/')[-1].split('.mp4')[0]
            self.num_frames[train] = frame_info[key]
        self.videos = {}
        if evaluate_video:
            self.valid, self.t_beg,self.num_clips = self._slide_windows(self.train,self.stride)
            i=0
            for data_index in self.train:
                frame_ix = self.num_frames[data_index]
                self.videos[data_index]=self._get_single_video(i,data_index,range(frame_ix))
                self.videos[data_index]['rgb']=im_to_video(self.videos[data_index]['rgb'])
                i+=1

        VideoDataset.__init__(self)

    def _set_datasetname(self):
        self.datasetname = "H2S"

    def _get_video_file(self, ind):
        return os.path.join(self.videos[ind])

    def _get_sequence(self, ind):
        return self.classes[ind], len(self.classes[ind])

    def _get_class(self, ind, frame_ix):
        total_duration = self.num_frames[ind]
        t_middle = frame_ix[0] + (self.num_in_frames / 2)
        # Uniformly distribute the glosses over the video
        # auto labels are only for training
        if (
            self.assign_labels == "uniform"
            or self.setname != "train"
            or len(self.frame_level_glosses[ind]) == 0
        ):
            glosses = self.classes[ind]
            num_glosses = len(glosses)
            duration_per_gloss = total_duration / num_glosses
            glossix = math.floor(t_middle / duration_per_gloss)
            return glosses[glossix]
        # Use the automatic alignments
        elif self.assign_labels == "auto":
            frame_glosses = self.frame_level_glosses[ind]
            lfg = len(frame_glosses)

            # LABEL OF THE MIDDLE FRAME
            # t_middle might fall out of boundary
            # in that case pick the last frame
            # if lfg <= int(t_middle):
            #     t_middle = lfg - 1
            # glossix = frame_glosses[int(t_middle)]

            # DOMINANT LABEL WITHIN THE CLIP
            clip_glosses = [
                frame_glosses[i]
                for i in frame_ix
                if i < lfg
            ]
            clip_glosses = np.asarray(clip_glosses)
            glss, cnts = np.unique(clip_glosses, return_counts=True)
            # If there are multiple max, choose randomly.
            max_indices = np.where(cnts == cnts.max())[0]
            selected_max_index = np.random.choice(max_indices)
            return glss[selected_max_index]
        else:
            exit()

    def _get_nframes(self, ind):
        return self.num_frames[ind]

    def _get_img_width(self, ind):
        return self.img_widths[ind]

    def _get_img_height(self, ind):
        return self.img_heights[ind]
