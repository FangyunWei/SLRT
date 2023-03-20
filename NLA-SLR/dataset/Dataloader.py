from dataset.VideoLoader import load_batch_video
from dataset.Dataset import build_dataset
import torch
from functools import partial
import random


def collate_fn_(batch, data_cfg, is_train, vocab, name2keypoint, word_emb_tab):
    outputs = {'names': [sample['name'] for sample in batch],
                'word_embs': None if word_emb_tab is None else torch.stack([torch.from_numpy(word_emb_tab[sample['label']]) for sample in batch], dim=0),
                'labels': [vocab.index(sample['label']) for sample in batch],
                'vlens': [sample['seq_len'] for sample in batch],
                'ori_video_files': [sample['video_file'] for sample in batch]
                }
    
    index_setting = data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])

    num_output_frames = data_cfg['num_output_frames']

    sgn_videos, sgn_keypoints = load_batch_video(
        zip_file = data_cfg['zip_file'], 
        names = outputs['names'], 
        vlens = outputs['vlens'], 
        dataset_name = data_cfg['dataset_name'], 
        is_train = is_train,
        num_output_frames = num_output_frames,
        name2keypoint = name2keypoint,
        index_setting=index_setting,
        temp_scale=data_cfg['transform_cfg'].get('temporal_augmentation', [1.0,1.0]),
        ori_video_files = outputs['ori_video_files'],
        from64=data_cfg['transform_cfg'].get('from64', False)  #sample 32 from 64
        )
    # print('hh')
    outputs['sgn_videos'] = sgn_videos
    outputs['sgn_keypoints'] = sgn_keypoints
    outputs['labels'] = torch.tensor(outputs['labels']).long()

    # A bug in lintel when load .mp4
    if 'WLASL' in data_cfg['dataset_name'] or 'SLR500' in data_cfg['dataset_name']:
        for i in range(len(outputs['vlens'])):
            outputs['vlens'][i] -= 2

    return outputs


def build_dataloader(cfg, split, is_train=True, val_distributed=False):
    dataset = build_dataset(cfg['data'], split)
    collate_func = partial(collate_fn_, data_cfg=cfg['data'], is_train=is_train, 
                            vocab=dataset.vocab, name2keypoint=dataset.name2keypoints, word_emb_tab=dataset.word_emb_tab)
    if is_train:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            shuffle=True
        )
    else:
        if val_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             collate_fn = collate_func,
                                             batch_size = cfg['training']['batch_size'],
                                             num_workers = cfg['training'].get('num_workers',2),
                                             sampler = sampler
                                             )
    return dataloader, sampler