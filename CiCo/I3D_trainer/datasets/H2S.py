import math
import os
import pickle as pkl

import cv2
import numpy as np

from datasets.videodataset import VideoDataset

cv2.setNumThreads(0)

import json
def load_dict(filename):
    '''load dict from json file'''
    with open(filename, "r") as json_file:
        dic = json.load(json_file)
    return dic

class H2S(VideoDataset):
    def __init__(
        self,
        inp_res=224,
        resize_res=256,
        setname="train",
        scale_factor=0.1,
        num_in_frames=16,
        evaluate_video=False,
        hflip=0.5,
        stride=1,
        gpu_collation=False,
        assign_labels="auto",
        rank=0,
    ):
        self.setname=setname
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

        file = open('misc/H2S/class.txt')
        class_dict = []
        line = file.readline().strip()
        label, class_name = line.split(' ')
        class_dict.append(class_name)
        while line and line != '':
            line = file.readline().strip()
            if line == '':
                break
            label, class_name = line.split(' ')
            class_dict.append(class_name)

        self.json_info=load_dict('misc/H2S/train_test_info.json')
        self.train = list(np.where(np.asarray(self.json_info["split"]) == 'train')[0])
        self.valid = list(np.where(np.asarray(self.json_info["split"]) == 'val')[0])
        self.num_frames=self.json_info["frame"]
        self.videos=self.json_info["video_path"]
        self.classes=[int(item_class_lable) for item_class_lable in self.json_info["class_label"]]
        self.class_names=class_dict
        if evaluate_video:
            self.valid, self.t_beg = self._slide_windows(self.valid)


        VideoDataset.__init__(self)

    def _set_datasetname(self):
        self.datasetname = "H2S"

    def _get_video_file(self, ind):
        return self.videos[ind]

    def _get_sequence(self, ind):
        return self.classes[ind], len(self.classes[ind])

    def _get_class(self, ind):
        return self.classes[ind]


    def _get_nframes(self, ind):
        return self.num_frames[ind]

    def _get_img_width(self, ind):
        return self.img_widths[ind]

    def _get_img_height(self, ind):
        return self.img_heights[ind]
