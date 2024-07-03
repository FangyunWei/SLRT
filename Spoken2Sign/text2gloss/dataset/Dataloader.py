from dataset.Sampler import MultiData_DistributedSampler
from dataset.VideoLoader import load_batch_video
from dataset.FeatureLoader import load_batch_feature
from dataset.Dataset import MixedDataset, build_dataset
import torch
import torch.nn.functional as F
from utils.misc import DATASETS
from collections import OrderedDict
import random
import numpy as np

def collate_fn_(inputs, data_cfg, task, is_train, dataset,
    text_tokenizer=None, gloss_tokenizer=None):
    name2keypoint = dataset.name2keypoints
    name2keypoint_extra = dataset.name2keypoints_extra
    memory_bank = dataset.memory_bank
    outputs = {
        'name':[i['name'] for i,n in inputs],
        'gloss':[i.get('gloss','') for i,n in inputs],
        'text':[i.get('text','') for i,n in inputs],
        'num_frames':[i.get('num_frames',None) for i,n in inputs],
        'datasetname': [n for i,n in inputs]}
    assert len(set(outputs['datasetname']))==1 #single source from a batch
    outputs['datasetname'] = outputs['datasetname'][0] # str
    if task == 'S2G' or (task=='S2T_glsfree' and data_cfg['input_data']=='video'):
        sgn_videos, sgn_keypoints, sgn_lengths, selected_indexs, pseudo_outputs = load_batch_video(
            zip_file=data_cfg['zip_file'], 
            names=outputs['name'], 
            num_frames=outputs['num_frames'], 
            transform_cfg=data_cfg['transform_cfg'], 
            dataset_name=data_cfg['dataset_name'], 
            pad_length=data_cfg.get('pad_length','pad_to_max'),
            pad = data_cfg.get('pad','replicate'),
            is_train=is_train,  #config for heatmap__loader
            name2keypoint=name2keypoint,
            memory_bank=memory_bank,
            name_sequences=[i.get('name_sequence',None) for i,n in inputs],
            boundary_sequences=[i.get('boundary_sequence',None) for i,n in inputs],
            gloss_length_distribution = [dataset.gloss_length_mean, dataset.gloss_length_std],
            pseudo_cfg=data_cfg.get('pseudo_cfg',{}),
            max_num_frames=data_cfg['max_sent_length'],
            need_video=('rgb' in data_cfg['input_streams']),
            name2keypoint_extra=name2keypoint_extra
            )
        if outputs['name'][0]=='pseudo':
            outputs['gloss'] = pseudo_outputs['gloss']
        if task!='S2T_glsfree' and gloss_tokenizer is not None:
            outputs['recognition_inputs'] = gloss_tokenizer(
                outputs['gloss'], outputs['datasetname'],
                pretokenized=(outputs['name'][0]=='pseudo'))
        else:
            outputs['recognition_inputs'] = {}
        #pseudo return gloss {'gls_lengths':gls_lengths, 'gloss_labels': batch_gls_ids}

        outputs['recognition_inputs']['sgn_videos'] = sgn_videos
        outputs['recognition_inputs']['sgn_keypoints'] = sgn_keypoints
        outputs['recognition_inputs']['sgn_lengths'] = sgn_lengths
        outputs['recognition_inputs']['datasetname'] = outputs['datasetname']
        outputs['recognition_inputs']['selected_indexs'] = selected_indexs
        outputs['recognition_inputs']['name'] = outputs['name']

    if task in ['S2T','G2T','S2T_glsfree','S2T_Ensemble','T2G']:
        tokenized_text = text_tokenizer(input_str=outputs['text'], text_input=(task=='T2G'))
        outputs['translation_inputs'] = {**tokenized_text}
        if task == 'S2T' or (task=='S2T_glsfree' and data_cfg['input_data']=='feature'):
            if not task == 'S2T_glsfree':
                outputs['recognition_inputs'] = gloss_tokenizer(outputs['gloss'], datasetname=outputs['datasetname'][0])
                #add for gloss+feature(translation_network input_type)
                outputs['translation_inputs']['gloss_ids'] = outputs['recognition_inputs']['gloss_labels']  
                outputs['translation_inputs']['gloss_lengths'] = outputs['recognition_inputs']['gls_lengths'] 
            else:
                outputs['recognition_inputs'] = {}
            for feature_name in ['head_rgb_input','head_keypoint_input']:
                # print(inputs[0])
                if feature_name in inputs[0][0]:
                    outputs['recognition_inputs'][feature_name], sgn_mask, sgn_lengths = \
                        load_batch_feature(features=[i[0][feature_name]+1.0e-8 for i in inputs])
            outputs['recognition_inputs']['sgn_mask'] = sgn_mask
            outputs['recognition_inputs']['sgn_lengths'] = sgn_lengths
            outputs['recognition_inputs']['datasetname'] = outputs['datasetname']
        elif task == 'G2T':
            tokenized_gloss = gloss_tokenizer(batch_gls_seq=outputs['gloss'])
            outputs['translation_inputs']['input_ids'] = tokenized_gloss['input_ids']
            outputs['translation_inputs']['attention_mask'] = tokenized_gloss['attention_mask']
        elif task == 'S2T_Ensemble':
            #inputs_embeds
            outputs['translation_inputs']['inputs_embeds_list'] = []
            outputs['translation_inputs']['attention_mask_list'] = []
            for ii in range(len(inputs[0][0]['inputs_embeds_list'])):
                inputs_embeds, mask_ ,_= \
                    load_batch_feature(features=[i[0]['inputs_embeds_list'][ii] for i in inputs])
                #mask_ =? attention_mask
                outputs['translation_inputs']['inputs_embeds_list'].append(inputs_embeds)
                outputs['translation_inputs']['attention_mask_list'].append(mask_)
        elif task == 'T2G':
            gls_tok_results = gloss_tokenizer(label_gls_seq=outputs['gloss'], need_input=False, need_label=True)
            # gls_tok_results = gloss_tokenizer(input_str=outputs['gloss'], text_input=False)
            outputs['translation_inputs']['labels'] = gls_tok_results['labels']
            outputs['translation_inputs']['decoder_input_ids'] = gls_tok_results['decoder_input_ids']
            # print(outputs['translation_inputs']['labels'], outputs['translation_inputs']['decoder_input_ids'])

    # combine sliding window features
    if 'sgn_features' in inputs[0][0]:
        fea_sample = data_cfg.get('fea_sample', 'avgpool')
        rgb_fea, pose_fea = [], []
        if task in ['S2T','G2T','S2T_glsfree','S2T_Ensemble']:
            selected_indexs = [np.arange(item[0]['num_frames']) if item[0]['num_frames']<=400 else np.arange(400) for item in inputs]
        for item, idx in zip(inputs, selected_indexs):
            # print(item[0]['sgn_features']['rgb'].shape)
            if fea_sample == 'stride':
                idx = idx[::4]
            r_fea = item[0]['sgn_features']['rgb']
            p_fea = item[0]['sgn_features']['keypoint']

            if 'upsample' in fea_sample:
                #upsample 4x
                num_frames = item[0]['num_frames']
                r_fea, p_fea = r_fea.unsqueeze(0).transpose(1,2), p_fea.unsqueeze(0).transpose(1,2)  #B,C,T
                r_fea, p_fea = F.interpolate(r_fea, size=num_frames, mode='linear'), F.interpolate(p_fea, size=num_frames, mode='linear')
                r_fea, p_fea = r_fea.squeeze(0).transpose(0,1), p_fea.squeeze(0).transpose(0,1)

            r_fea = r_fea[idx]
            p_fea = p_fea[idx]
            if 'noise' in fea_sample:
                # random noise as feature
                r_fea = torch.randn_like(r_fea)
                p_fea = torch.randn_like(p_fea)

            swap_ratio_and_prob = data_cfg.get('swap_ratio_and_prob', '2_0.5')
            swap_ratio, prob = int(swap_ratio_and_prob.split('_')[0]), float(swap_ratio_and_prob.split('_')[1])
            if is_train and 'aug' in fea_sample and random.random() < prob:
                #feature augmentation
                #select indexes
                random.shuffle(idx)
                num_swap = max(len(idx)//swap_ratio, 1)
                idx_swap = sorted(idx[:num_swap])
                idx = sorted(idx)
                for i_s in idx_swap:
                    # get label
                    vfile = item[0]['name']
                    label = dataset.vfile2seq[vfile][i_s]
                    if label != '<blank>':
                        try:
                            # get ratio (proper position from a clip)
                            for candidate in dataset.vfile2pos[vfile][label]:
                                if i_s in candidate:
                                    ratio = (i_s-candidate[0])/len(candidate)
                            fea_dict = random.sample(dataset.label2fea[label], 1)[0]
                            r_fea_swap, p_fea_swap = fea_dict['rgb'], fea_dict['keypoint']
                            sel = int(r_fea_swap.shape[0] * ratio)
                            r_fea_swap, p_fea_swap = r_fea_swap[sel], p_fea_swap[sel]
                            idx_rec = idx.index(i_s)
                            r_fea[idx_rec] = r_fea_swap
                            p_fea[idx_rec] = p_fea_swap
                            # print(ratio)
                        except:
                            continue

            rgb_fea.append(r_fea)
            pose_fea.append(p_fea)

        if task in ['S2T','G2T','S2T_glsfree','S2T_Ensemble']:
            batch_rgb_fea, _, _ = load_batch_feature(rgb_fea)
            batch_pose_fea, _, _ = load_batch_feature(pose_fea)
            outputs['recognition_inputs']['sgn_features'] = [F.avg_pool1d(batch_rgb_fea.permute(0,2,1), kernel_size=4, stride=4).permute(0,2,1), 
                                                            F.avg_pool1d(batch_pose_fea.permute(0,2,1), kernel_size=4, stride=4).permute(0,2,1)]

        else:
            if fea_sample in ['stride']:
                outputs['recognition_inputs']['sgn_features'] = [torch.stack(rgb_fea, dim=0), torch.stack(pose_fea, dim=0)]
            elif 'avgpool' in fea_sample:
                outputs['recognition_inputs']['sgn_features'] = [F.avg_pool1d(torch.stack(rgb_fea, dim=0).permute(0,2,1), kernel_size=4, stride=4).permute(0,2,1), 
                                                                F.avg_pool1d(torch.stack(pose_fea, dim=0).permute(0,2,1), kernel_size=4, stride=4).permute(0,2,1)]
            elif 'maxpool' in fea_sample:
                outputs['recognition_inputs']['sgn_features'] = [F.max_pool1d(torch.stack(rgb_fea, dim=0).permute(0,2,1), kernel_size=4, stride=4).permute(0,2,1), 
                                                            F.max_pool1d(torch.stack(pose_fea, dim=0).permute(0,2,1), kernel_size=4, stride=4).permute(0,2,1)]
    elif 'recognition_inputs' in outputs:
        outputs['recognition_inputs']['sgn_features'] = [None, None]

    return outputs

def build_dataloader(cfg, split, 
    text_tokenizer=None, gloss_tokenizer=None, 
    mode='auto', val_distributed=False):
    if cfg['data'].get('multi',False)==True:
        dataset_collect = OrderedDict()
        # assert cfg['training']['batch_size']==1
        for datasetname in sorted(DATASETS):
            if datasetname in cfg['data']:
                dataset_collect[datasetname] = build_dataset(cfg['data'][datasetname], split)
        dataset = MixedDataset(dataset_collect)
        collate_fn = lambda x:collate_fn_(
                            inputs=x, task=cfg['task'], 
                            data_cfg=cfg['data'][x[0][1]],
                            is_train=(mode=='train'),
                            text_tokenizer=text_tokenizer,
                            gloss_tokenizer=gloss_tokenizer,
                            dataset=dataset.datasets[x[0][1]])
    else:
        raise ValueError #already modifed in misc.py load_config
        dataset = build_dataset(cfg['data'], split)
        collate_fn = lambda x:collate_fn_(
                            inputs=x, task=cfg['task'], 
                            data_cfg=cfg['data'][x[0][1]],
                            is_train=(mode=='train'),
                            text_tokenizer=text_tokenizer,
                            gloss_tokenizer=gloss_tokenizer,
                            name2keypoint=dataset_collect[x[0][1]].name2keypoints)
    mode = split if mode=='auto' else mode
    if mode=='train':
        if 'RecognitionNetwork' in cfg['model'] and cfg['model']['RecognitionNetwork'].get('multidata_sampler',False)==True:
            sampler = MultiData_DistributedSampler(
                name2dataset=dataset_collect, 
                shuffle=cfg['training']['shuffle'] and split=='train')
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, 
                shuffle=cfg['training']['shuffle'] and split=='train'
            )
    else:
        if val_distributed:
            if 'RecognitionNetwork' in cfg['model'] and cfg['model']['RecognitionNetwork'].get('multidata_sampler',False)==True:            
                sampler = MultiData_DistributedSampler(
                    name2dataset=dataset_collect, 
                    shuffle=False) 
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)           
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            collate_fn=collate_fn,
                                            batch_size=cfg['training']['batch_size'],
                                            num_workers=cfg['training'].get('num_workers',2),
                                            sampler=sampler,
                                            )
    return dataloader, sampler