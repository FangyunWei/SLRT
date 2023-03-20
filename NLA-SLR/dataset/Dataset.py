import torch, pickle
import json, os, gzip
from glob import glob
import numpy as np
from utils.misc import get_logger


Hrnet_Part2index = {
    'pose': list(range(11)),
    'hand': list(range(91, 133)),
    'mouth': list(range(71,91)),
    'face_others': list(range(23, 71))
}
for k_ in ['mouth','face_others', 'hand']:
    Hrnet_Part2index[k_+'_half'] = Hrnet_Part2index[k_][::2]
    Hrnet_Part2index[k_+'_1_3'] = Hrnet_Part2index[k_][::3]
    
def get_keypoints_num(keypoint_file, use_keypoints):
    keypoints_num = 0
    assert 'hrnet' in keypoint_file
    Part2index = Hrnet_Part2index
    for k in sorted(use_keypoints):
        keypoints_num += len(Part2index[k])     
    return keypoints_num


class ISLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg, split):
        super(ISLRDataset, self).__init__()
        self.split = split #train, dev, test
        self.dataset_cfg = dataset_cfg
        self.root = os.path.join(*(self.dataset_cfg[split].split('/')[:-1]))
        if 'MSASL' in dataset_cfg['dataset_name']:
            self.vocab = self.create_vocab()
            self.annotation = self.load_annotations(split)
        else:
            self.annotation = self.load_annotations(split)
            self.vocab = self.create_vocab()
        # print(len(self.vocab))
        self.input_streams = dataset_cfg.get('input_streams', ['rgb'])
        self.logger = get_logger()
        self.name2keypoints = self.load_keypoints()
        self.word_emb_tab = None
        if dataset_cfg.get('word_emb_file', None):
            self.word_emb_tab = self.load_word_emb_tab()

    def load_keypoints(self):
        if 'keypoint' in self.input_streams or 'keypoint_coord' in self.input_streams or 'trajectory' in self.input_streams:
            with open(self.dataset_cfg['keypoint_file'],'rb') as f:
                name2all_keypoints = pickle.load(f)
            assert 'hrnet' in self.dataset_cfg['keypoint_file']
            self.logger.info('Keypoints source: hrnet')
            Part2index = Hrnet_Part2index

            name2keypoints = {}
            for name, all_keypoints in name2all_keypoints.items():
                name2keypoints[name] = []
                for k in sorted(self.dataset_cfg['use_keypoints']):
                    selected_index = Part2index[k]
                    name2keypoints[name].append(all_keypoints[:, selected_index]) # T, N, 3
                name2keypoints[name] = np.concatenate(name2keypoints[name], axis=1) #T, N, 3
                self.keypoints_num = name2keypoints[name].shape[1]
            
            self.logger.info(f'Total #={self.keypoints_num}') 
            assert self.keypoints_num == get_keypoints_num(self.dataset_cfg['keypoint_file'], self.dataset_cfg['use_keypoints'])
        
        else:
            name2keypoints = None
        return name2keypoints

    def load_annotations(self, split):
        self.annotation_file = self.dataset_cfg[split]
        self.root = os.path.join(*(self.annotation_file.split('/')[:-1]))
        try:
            with open(self.annotation_file, 'rb') as f:
                annotation = pickle.load(f)
        except:
            with gzip.open(self.annotation_file, 'rb') as f:
                annotation = pickle.load(f)
        
        # clean WLASL
        if 'WLASL' in self.dataset_cfg['dataset_name']:
            variant_file = self.dataset_cfg['dataset_name'].split('_')[-1]+'.json'
            variant_file = os.path.join(self.root, variant_file)
            with open(variant_file, 'r') as f:
                variant = json.load(f)
            cleaned = []
            for item in annotation:
                if 'augmentation' not in item['video_file'] and item['name'] in list(variant.keys()):
                    cleaned.append(item)
            annotation = cleaned

        elif 'MSASL' in self.dataset_cfg['dataset_name']:
            # num = int(self.dataset_cfg['dataset_name'].split('_')[-1])
            cleaned = []
            for item in annotation:
                if item['label'] in self.vocab:
                    cleaned.append(item)
            annotation = cleaned
        return annotation
    
    def load_word_emb_tab(self):
        fname = self.dataset_cfg['word_emb_file']
        with open(fname, 'rb') as f:
            word_emb_tab = pickle.load(f)
        return word_emb_tab
    
    def create_vocab(self):
        if 'WLASL' in self.dataset_cfg['dataset_name'] or 'NMFs-CSL' in self.dataset_cfg['dataset_name']:
            annotation = self.load_annotations('train')
            vocab = []
            for item in annotation:
                if item['label'] not in vocab:
                    vocab.append(item['label'])
            vocab = sorted(vocab)
        elif 'MSASL' in self.dataset_cfg['dataset_name']:
            with open(os.path.join(self.root, 'MSASL_classes.json'), 'rb') as f:
                all_vocab = json.load(f)
            num = int(self.dataset_cfg['dataset_name'].split('_')[-1])
            vocab = all_vocab[:num]
        return vocab

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        return self.annotation[idx]


def build_dataset(dataset_cfg, split):
    dataset = ISLRDataset(dataset_cfg, split)
    return dataset