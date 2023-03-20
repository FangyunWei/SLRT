from tkinter.filedialog import Open
import torch, pickle
import gzip, os
from glob import glob
import numpy as np
from utils.misc import get_logger

Openpose_Part2index = {
    'pose': [('pose', [0,1,2,3,4,5,6,7,14,15,16,17])],
    'mouth': [('face', list(range(48, 68)))],
    'face_others': [('face', list(range(0,48))+list(range(68,70)))],
    'hand': [('hand_0', list(range(21))),('hand_1', list(range(21)))]
}
Hrnet_Part2index = {
    'pose': [('keypoints', list(range(11)))],
    'hand': [('keypoints', list(range(91, 133)))],
    'mouth': [('keypoints', list(range(71,91)))],
    'face_others': [('keypoints', list(range(23, 71)))]
}
for k_ in ['mouth','face_others', 'hand']:
    Openpose_Part2index[k_+'_half'] = [(d[0], d[1][::2]) for d in Openpose_Part2index[k_]]
    Hrnet_Part2index[k_+'_half'] = [(d[0], d[1][::2]) for d in Hrnet_Part2index[k_]]
    Openpose_Part2index[k_+'_1_3'] = [(d[0], d[1][::3]) for d in Openpose_Part2index[k_]]
    Hrnet_Part2index[k_+'_1_3'] = [(d[0], d[1][::3]) for d in Hrnet_Part2index[k_]]
    
def get_keypoints_num(keypoint_file, use_keypoints):
    keypoints_num = 0
    Part2index = Hrnet_Part2index
    for k in sorted(use_keypoints):
        keypoints_num += sum([len(index) for key_, index in Part2index[k]])     
    return keypoints_num

class SignLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg, split):
        super(SignLanguageDataset, self).__init__()
        self.split = split #train, val, test
        self.dataset_cfg = dataset_cfg
        self.load_annotations()
        self.input_streams = dataset_cfg.get('input_streams',['rgb'])
        self.logger = get_logger()
        self.load_keypoints()

    def load_keypoints(self):
        if 'keypoint' in self.input_streams:
            with open(self.dataset_cfg['keypoint_file'],'rb') as f:
                name2all_keypoints = pickle.load(f)
            Part2index = Hrnet_Part2index
            self.name2keypoints = {}
            for name, all_keypoints in name2all_keypoints.items():
                self.name2keypoints[name] = []
                for k in sorted(self.dataset_cfg['use_keypoints']):
                    for key_, selected_index in Part2index[k]:
                        self.name2keypoints[name].append(all_keypoints[key_][:,selected_index]) # T, N, 3
                self.name2keypoints[name] = torch.tensor(np.concatenate(self.name2keypoints[name], axis=1)) #T, N, 3
                self.keypoints_num = self.name2keypoints[name].shape[1]
                if len(self.name2keypoints)==1:
                    self.logger.info(f'Total #={self.keypoints_num}') 
            assert self.keypoints_num==get_keypoints_num(self.dataset_cfg['keypoint_file'], self.dataset_cfg['use_keypoints'])
        else:
            self.name2keypoints = None

    def load_annotations(self):
        self.annotation_file = self.dataset_cfg[self.split]
        with gzip.open(self.annotation_file, 'rb') as f:
            self.annotation = pickle.load(f)
        for a in self.annotation:
            a['sign_features'] = a.pop('sign',None)

        for feature_name in ['head_rgb_input','head_keypoint_input']:
            filename = self.dataset_cfg.get(self.split+f'_{feature_name}','')
            if os.path.isfile(filename):
                with gzip.open(filename, 'rb') as f:
                    annotation = pickle.load(f)
                name2feature = {a['name']:a['sign'] for a in annotation}
                for a in self.annotation:
                    a[feature_name] = name2feature[a['name']]

        if f'{self.split}_inputs_embeds' in self.dataset_cfg:
            for a in self.annotation:
                a['inputs_embeds_list'] = []
            for filename in self.dataset_cfg[f'{self.split}_inputs_embeds']:
                with gzip.open(filename, 'rb') as f:
                    annotation = pickle.load(f)
                name2feature = {a['name']:a['sign'] for a in annotation}
                for a in self.annotation:
                    a['inputs_embeds_list'].append(name2feature[a['name']])

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        return {k:v for k, v in self.annotation[idx].items() \
            if k in [
                'name','gloss','text','num_frames','sign',
                'head_rgb_input','head_keypoint_input',
                'inputs_embeds_list']}

# class Gloss2TextDataset(SignLanguageDataset):
#     def __init__(self, dataset_cfg, split):
#         super().__init__(dataset_cfg, split)
#     # def __getitem__(self, idx):

def build_dataset(dataset_cfg, split):
    dataset = SignLanguageDataset(dataset_cfg, split)
    return dataset