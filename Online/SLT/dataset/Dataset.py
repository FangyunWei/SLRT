from tkinter.filedialog import Open
import torch, pickle
import gzip, os
from glob import glob
import numpy as np
from collections import defaultdict
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
for k_ in ['mouth','face_others', 'hand', 'pose']:
    Openpose_Part2index[k_+'_half'] = [(d[0], d[1][::2]) for d in Openpose_Part2index[k_]]
    Hrnet_Part2index[k_+'_half'] = [(d[0], d[1][::2]) for d in Hrnet_Part2index[k_]]
    Openpose_Part2index[k_+'_1_3'] = [(d[0], d[1][::3]) for d in Openpose_Part2index[k_]]
    Hrnet_Part2index[k_+'_1_3'] = [(d[0], d[1][::3]) for d in Hrnet_Part2index[k_]]
    Hrnet_Part2index[k_+'_1_4'] = [(d[0], d[1][::4]) for d in Hrnet_Part2index[k_]]
    Hrnet_Part2index[k_+'_1_6'] = [(d[0], d[1][::6]) for d in Hrnet_Part2index[k_]]
    
def get_keypoints_num(keypoint_file, use_keypoints):
    keypoints_num = 0
    if 'openpose' in keypoint_file:
        Part2index = Openpose_Part2index
    elif 'hrnet' in keypoint_file:
        Part2index = Hrnet_Part2index
    else:
        raise ValueError
    for k in sorted(use_keypoints):
        keypoints_num += sum([len(index) for key_, index in Part2index[k]])     
    return keypoints_num

class SignLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg, split):
        super(SignLanguageDataset, self).__init__()
        self.logger = get_logger()
        self.split = split #train, val, test
        self.dataset_cfg = dataset_cfg
        self.load_annotations()
        self.input_streams = dataset_cfg.get('input_streams',['rgb'])
        self.load_keypoints()
        self.pseudo, self.memory_bank = [], []

        self.load_pseudo_val()

    def load_pseudo_val(self):
        self.name2pseudo_val = None
        


    def load_keypoints(self):
        if 'keypoint' in self.input_streams:
            with open(self.dataset_cfg['keypoint_file'],'rb') as f:
                name2all_keypoints = pickle.load(f)
            if 'openpose' in self.dataset_cfg['keypoint_file']:
                self.logger.info('Keypoints source: openpose')
                Part2index = Openpose_Part2index
            elif 'hrnet' in self.dataset_cfg['keypoint_file']:
                self.logger.info('Keypoints source: hrnet')
                Part2index = Hrnet_Part2index
            else:
                raise ValueError

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
        try:
            with open(self.annotation_file, 'rb') as f:
                self.annotation = pickle.load(f)
        except:
            with gzip.open(self.annotation_file, 'rb') as f:
                self.annotation = pickle.load(f)
        gloss_length = []
        for a in self.annotation:
            a['sign_features'] = a.pop('sign',None)
            gloss_length.append(len(a['gloss'].split()))
        self.gloss_length_mean = np.mean(gloss_length)
        self.gloss_length_std = np.std(gloss_length)
        self.logger.info('{} gloss_length {:.2f}+_{:.1f}'.format(self.annotation_file, self.gloss_length_mean, self.gloss_length_std))

        for feature_name in ['head_rgb_input','head_keypoint_input']:
            m = feature_name.split('_')[1]
            filename = self.dataset_cfg.get(self.split+f'_{feature_name}','')
            # print(m)
            if os.path.isfile(filename):
                if 'extract_feature' in filename:
                    with gzip.open(filename, 'rb') as f:
                        annotation = pickle.load(f)
                    name2feature = {a['name']:a['sign'] for a in annotation}
                else:
                    with open(filename, 'rb') as f:
                        annotation = pickle.load(f)
                    if 's2g' in filename:
                        name2feature = annotation[m]
                    else:
                        name2feature = annotation[f'{m}_blk5']
                for a in self.annotation:
                    a[feature_name] = name2feature[a['name']]
        
        if f'{self.split}_sgn_features' in self.dataset_cfg:
            fea_sample = self.dataset_cfg.get('fea_sample', 'stride')
            with open(self.dataset_cfg[f'{self.split}_sgn_features'], 'rb') as f:
                annotation = pickle.load(f)
            iso_ann = None
            if self.split == 'train' and 'aug' in fea_sample:
                with open(self.dataset_cfg['iso_file'], 'rb') as f:
                    iso_ann = pickle.load(f)
                vfile2len = {}
                self.vfile2seq = defaultdict(list)
                for item in self.annotation:
                    vfile2len[item['name']] = item['num_frames']
                    self.vfile2seq[item['name']] = ['<blank>'] * item['num_frames']
                self.vfile2pos = {}
                self.label2fea = defaultdict(list)
                for item in iso_ann:
                    vfile = item['video_file']
                    label = item['label']
                    start, end = item['start'], item['end']
                    num_frames = vfile2len[vfile]
                    for i in range(num_frames):
                        if i>=start and i<end:
                            self.vfile2seq[vfile][i] = label
                    if vfile not in self.vfile2pos:
                        self.vfile2pos[vfile] = defaultdict(list)
                    self.vfile2pos[vfile][label].append([_ for _ in range(start, end)])
                    self.label2fea[label].append({'rgb': annotation['rgb'][vfile][start:end], 'keypoint': annotation['keypoint'][vfile][start:end]})
            for a in self.annotation:
                a['sgn_features'] = {'rgb': annotation['rgb'][a['name']], 'keypoint': annotation['keypoint'][a['name']]}

        if f'{self.split}_inputs_embeds' in self.dataset_cfg:
            for a in self.annotation:
                a['inputs_embeds_list'] = []
            for filename in self.dataset_cfg[f'{self.split}_inputs_embeds']:
                with gzip.open(filename, 'rb') as f:
                    annotation = pickle.load(f)
                name2feature = {a['name']:a['sign'] for a in annotation}
                for a in self.annotation:
                    a['inputs_embeds_list'].append(name2feature[a['name']])

    def set_pseudo(self, ratio, memory_bank):
        n_pseudo = int(ratio*len(self.annotation))
        self.pseudo = [{'name':'pseudo'} for i in range(n_pseudo)]
        self.memory_bank = memory_bank
        self.logger.info(f'{self.dataset_cfg["dataset_name"]} #pseudo={n_pseudo} ({ratio}x{len(self.annotation)})')
        self.logger.info(f'Using memory bank, #vocab={len(memory_bank)}')

    def __len__(self):
        return len(self.annotation)+len(self.pseudo)
    
    def __getitem__(self, idx):
        if idx < len(self.annotation):
            return {k:v for k, v in self.annotation[idx].items() \
                if k in [
                    'name','gloss','text','num_frames','sign',
                    'head_rgb_input','head_keypoint_input', 'sgn_features',
                    'inputs_embeds_list',
                    'name_sequence', 'boundary_sequence']}, self.dataset_cfg['dataset_name']
        else:
            return self.pseudo[idx-len(self.annotation)], self.dataset_cfg['dataset_name']

# class Gloss2TextDataset(SignLanguageDataset):
#     def __init__(self, dataset_cfg, split):
#         super().__init__(dataset_cfg, split)
#     # def __getitem__(self, idx):

def build_dataset(dataset_cfg, split):
    dataset = SignLanguageDataset(dataset_cfg, split)
    return dataset



class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets) -> None:
        super().__init__()
        self.datasets = datasets #dicts
        self.logger = get_logger()
        self.logger.info('Merged Datasets:')
        for d in datasets: #ordered dict
            self.logger.info('{}:{}'.format(d,len(datasets[d])))
        self.nums = [len(datasets[d]) for d in datasets]
        self.total_num = sum(self.nums)
        self.index2datasets = []
        for name in self.datasets:
            for j in range(len(self.datasets[name])):
                self.index2datasets.append([name, j])

    def set_pseudo(self, ratio, dataloader, memory_bank):
        for d in self.datasets:
            self.datasets[d].set_pseudo(ratio, memory_bank[d])
        self.__init__(self.datasets)
        assert type(dataloader.sampler)==torch.utils.data.distributed.DistributedSampler, type(dataloader.sampler)
        new_sampler = torch.utils.data.distributed.DistributedSampler(
                self, shuffle=dataloader.sampler.shuffle, seed=dataloader.sampler.seed, drop_last=dataloader.sampler.drop_last)
        new_sampler.set_epoch(dataloader.sampler.epoch)
        new_dataloader = torch.utils.data.DataLoader(self,
                                            collate_fn=dataloader.collate_fn,
                                            batch_size=dataloader.batch_size,
                                            num_workers=dataloader.num_workers,
                                            sampler=new_sampler,
                                            )
        return new_dataloader, new_sampler

    def __len__(self):
        return self.total_num #sum over all datasets
    
    def __getitem__(self, index):
        dataset_name, dataset_index = self.index2datasets[index]
        return self.datasets[dataset_name][dataset_index]#, dataset_name already included in the 1st item