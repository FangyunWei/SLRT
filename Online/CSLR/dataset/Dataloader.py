from dataset.VideoLoader import load_batch_video
from dataset.FeatureLoader import load_batch_feature
from dataset.Dataset import build_dataset
import torch
import torch.nn.functional as F
from functools import partial
import random
import numpy as np


def pad_tensor(x, pad_left, pad_right):
    assert x.ndim in [2, 3, 4]
    if pad_left > 0:
        if x.ndim == 4:
            pad = x[0].repeat(pad_left, 1, 1, 1)
        elif x.ndim == 3:
            pad = x[0].repeat(pad_left, 1, 1)
        elif x.ndim == 2:
            pad = x[0].repeat(pad_left, 1)
        x = torch.cat([pad, x], dim=0)

    if pad_right > 0:
        if x.ndim == 4:
            pad = x[-1].repeat(pad_right, 1, 1, 1)
        elif x.ndim == 3:
            pad = x[-1].repeat(pad_right, 1, 1)
        elif x.ndim == 2:
            pad = x[-1].repeat(pad_right, 1)
        x = torch.cat([x, pad], dim=0)
    return x


def collate_fn_(batch, data_cfg, is_train, vocab, name2keypoint, word_emb_tab, vfile2raw_vlens, 
                task='ISLR', g2g_input_type='gloss', g2g_tokenizer=None, g2g_input=None, vfile2framelabel={}):
    use_bag = False
    if isinstance(batch[0], list):
        use_bag = True
        bag_size = data_cfg['transform_cfg'].get('bag_size', 6)
        base_first = data_cfg['transform_cfg'].get('bag_base_first', False)
        new_batch = []
        for item_lst in batch:
            idx = np.random.permutation(np.arange(len(item_lst)))
            if len(item_lst) >= bag_size:
                idx = idx[:bag_size]
            else:
                copied_idx = np.random.randint(0, len(item_lst), bag_size-len(item_lst))
                idx = np.concatenate([idx, copied_idx])
            if base_first and (idx==0).sum()<1:
                # there must be at lease one base sample
                idx[np.random.randint(idx.shape[0], size=1)[0]] = 0
            new_batch.extend([item_lst[i] for i in idx])
        batch = new_batch

    outputs = {'names': [sample['name'] for sample in batch],
                'word_embs': None if word_emb_tab is None else torch.stack([torch.from_numpy(word_emb_tab[sample['label']]) for sample in batch], dim=0),
                'labels': [vocab.index(sample['label']) for sample in batch] if data_cfg['dataset_name'] not in ['phoenix', 'phoenix2014', 'phoenixcomb', 'csl'] else [0 for sample in batch],
                # 'aug': [sample['aug'] for sample in batch] if data_cfg['dataset_name'] in ['phoenix_iso', 'phoenix_comb_iso'] and is_train else [0 for sample in batch],  #if IOU augmented
                'aug': [0 for sample in batch],
                'vlens': [sample['seq_len'] for sample in batch] if data_cfg['dataset_name'] not in ['phoenix', 'phoenix2014', 'phoenixcomb', 'csl'] else [sample['num_frames'] for sample in batch],
                # 'raw_vlens': [sample['raw_seq_len'] for sample in batch] if data_cfg['dataset_name'] in ['phoenix_comb_iso'] and is_train else [0 for sample in batch],
                'raw_vlens': [0 for sample in batch],
                'ori_video_files': [sample['video_file'] for sample in batch] if data_cfg['dataset_name'] not in ['phoenix', 'phoenix2014', 'phoenixcomb', 'csl'] else [sample['name'] for sample in batch],
                'gls_ref': [sample['gloss'] for sample in batch] if data_cfg['dataset_name'] in ['phoenix', 'phoenix2014', 'phoenixcomb', 'csl'] else None,
                'bag_labels': [sample['bag'] for sample in batch] if use_bag else None,
                'iou_labels': None,
                'temp_idx': None,
                'sgn_videos': None,
                'sgn_keypoints': None,
                'start_idx': None
                }
    
    if task == 'ISLR':
        num_output_frames = data_cfg['num_output_frames']

        if len(vfile2raw_vlens) > 0:
            outputs['raw_vlens'] = [vfile2raw_vlens[sample['video_file']] for sample in batch]

        if 'base_start' in batch[0]:
            # scale at T//2!
            outputs['temp_idx'] = []
            ratio = data_cfg.get('temp_idx_ratio', 2)  #if get temp_idx of original scale
            for sample in batch:
                start, end, vlen, base_start, base_end = sample['start'], sample['end'], sample['seq_len'], sample['base_start'], sample['base_end']
                if sample['aug'] == 0:
                    # for base data, take the center
                    if vlen < num_output_frames:
                        temp_start = (num_output_frames-vlen) // (2*ratio)
                        temp_end = temp_start + vlen//ratio
                    else:
                        temp_start, temp_end = 0, num_output_frames//ratio
                else:
                    temp_start = max(start, base_start) - start
                    temp_end = min(end, base_end) - start
                    temp_start, temp_end = temp_start//ratio, temp_end//ratio
                temp_start = max(0, temp_start)
                temp_start = min(temp_start, num_output_frames//ratio-1)
                temp_end = min(temp_end, num_output_frames//ratio)
                temp_end = max(temp_end, temp_start+1)
                outputs['temp_idx'].append([temp_start, temp_end])
            outputs['temp_idx'] = torch.tensor(outputs['temp_idx']).long()

        index_setting = data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
        sgn_videos, sgn_keypoints, start_idx = load_batch_video(
            zip_file = data_cfg['zip_file'], 
            names = outputs['names'], 
            vlens = outputs['vlens'], 
            raw_vlens = outputs['raw_vlens'],  #for sliding window only, it is the length of the raw continuous video
            dataset_name = data_cfg['dataset_name'], 
            is_train = is_train,
            num_output_frames = num_output_frames,
            name2keypoint = name2keypoint,
            index_setting=index_setting,
            temp_scale=data_cfg['transform_cfg'].get('temporal_augmentation', [1.0,1.0]),
            ori_video_files = outputs['ori_video_files'],
            fps=data_cfg['transform_cfg'].get('fps', 1),
            from64=data_cfg['transform_cfg'].get('from64', False)  #sample 32 from 64
            )
        # print('hh')
        outputs['sgn_videos'] = sgn_videos
        outputs['sgn_keypoints'] = sgn_keypoints
        # if ins_rep and is_train:
        #     outputs['names'] = outputs['names']*2
        #     outputs['labels'] = outputs['labels']*2
        #     outputs['vlens'] = outputs['vlens']*2
        outputs['labels'] = torch.tensor(outputs['labels']).long()
        outputs['aug'] = torch.tensor(outputs['aug']).long()
        if outputs['bag_labels'] is not None:
            outputs['bag_labels'] = torch.tensor(outputs['bag_labels']).long()

        # A bug in lintel when load .mp4
        if 'WLASL' in data_cfg['dataset_name'] or 'SLR500' in data_cfg['dataset_name']:
            for i in range(len(outputs['vlens'])):
                outputs['vlens'][i] -= 2
        outputs['start_idx'] = start_idx

    # generate G2G inputs
    if task == 'G2G':
        tokenized_label = g2g_tokenizer(label_gls_seq=outputs['gls_ref'], need_input=False, need_label=True)
        outputs['translation_inputs'] = {**tokenized_label}

        if g2g_input_type == 'gloss':
            label_gls_seq = outputs['gls_ref']
            input_gls_seq = []
            for name in outputs['names']:
                input_gls_seq.append(g2g_input[name]['window_greedy_7_raw_gls_hyp_top2'])
            blank_as_mask = data_cfg.get('blank_as_mask', True)
            outputs['translation_inputs'].update(g2g_tokenizer(input_gls_seq, label_gls_seq, blank_as_mask, need_input=True, need_label=False))
        
        elif g2g_input_type in ['feature', 'prob']:
            fea = []
            for name in outputs['names']:
                if g2g_input_type == 'feature':
                    fea.append(torch.cat([g2g_input['rgb_blk5'][name], g2g_input['keypoint_blk5'][name]], dim=-1))
                else:
                    fea.append(torch.from_numpy(g2g_input[name]).softmax(dim=-1))
            fea, _, lengths = load_batch_feature(fea)
            # print('input fea shape: ', fea.shape)
            outputs['translation_inputs'].update({'input_feature': fea, 'input_lengths': lengths})

    elif task == 'bag_denoise':
        fea = []
        labels = []
        for name in outputs['names']:
            fea.append(torch.cat([g2g_input['rgb_blk5'][name], g2g_input['keypoint_blk5'][name]], dim=-1))
            labels.extend(vfile2framelabel[name])
        fea = fea[0]  #T,C
        winsize = 7
        fea = pad_tensor(fea, winsize//2, winsize//2)
        fea = fea.unfold(0, winsize, 1).transpose(1,2)  #B,w,C
        #make labels
        labels = torch.tensor([vocab.index(i) for i in labels]).long()
        assert len(labels) == fea.shape[0]
        outputs['denoise_inputs'] = {'features': fea, 'labels': labels}

    return outputs


def build_dataloader(cfg, split, task='ISLR', g2g_tokenizer=None, is_train=True, val_distributed=False):
    dataset = build_dataset(cfg['data'], split, task)
    collate_func = partial(collate_fn_, data_cfg=cfg['data'], is_train=is_train, 
                            vocab=dataset.vocab, name2keypoint=dataset.name2keypoints, 
                            word_emb_tab=dataset.word_emb_tab, vfile2raw_vlens=dataset.vfile2raw_vlens,
                            task=task, g2g_input_type=cfg['model']['TranslationNetwork']['input_type'] if 'TranslationNetwork' in cfg['model'] else 'gloss', 
                            g2g_tokenizer=g2g_tokenizer, g2g_input=dataset.g2g_input, vfile2framelabel=dataset.vfile2framelabel)
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