from turtle import forward
import warnings, wandb
import pickle
from collections import defaultdict
from modelling.model import build_model
from utils.optimizer import build_optimizer, build_scheduler
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import os, sys
import shutil
import queue
import json, math
sys.path.append(os.getcwd())#slt dir
import torch
from torch.nn.parallel import DistributedDataParallel as DDP, distributed
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from utils.misc import (
    get_logger,
    load_config,
    log_cfg,
    load_checkpoint,
    make_logger, make_writer,
    set_seed,
    symlink_update,
    is_main_process, init_DDP, move_to_device,
    neq_load_customized,
    synchronize,
)
from utils.metrics import compute_accuracy, wer_list
from utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans
from dataset.Dataloader import build_dataloader
from dataset.Dataset import build_dataset
from utils.progressbar import ProgressBar
from copy import deepcopy
from itertools import groupby
from functools import partial
import math


def map_phoenix_gls(g_lower):#lower->upper
    if 'neg-' in g_lower[:4]:
        g_upper = 'neg-'+g_lower[4:].upper()
    elif 'poss-' in g_lower:
        g_upper = 'poss-'+g_lower[5:].upper()
    elif 'negalp-' in g_lower:
        g_upper = 'negalp-'+g_lower[7:].upper()
    elif 'loc-' in g_lower:
        g_upper = 'loc-'+g_lower[7:].upper()
    elif 'cl-' in g_lower:
        g_upper = 'cl-'+g_lower[7:].upper()
    else:
        g_upper = g_lower.upper()
    return g_upper


def index2token(index, vocab, dataset_name='phoenix'):
    if dataset_name == 'phoenix':
        token = [map_phoenix_gls(vocab[i]) for i in index]
    else:
        token = [vocab[i] for i in index]
    # clean = []
    # # clean 2-gram
    # for t in token:
    #     if '_' in t:
    #         clean.extend(t.split('_'))
    #     else:
    #         clean.append(t)
    return token


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


def sliding_windows(video, keypoint, win_size=16, stride=1, save_fea=False):
    B, T = video.shape[:2]
    assert B==1
    video = video.squeeze(0)
    keypoint = keypoint.squeeze(0)

    num_clips = math.ceil(T/stride)
    num_clips = max(num_clips, 1)
    final_frames = (num_clips-1) * stride + win_size
    pad_left = (final_frames - T) // 2
    pad_right = final_frames - T - pad_left
    
    #pad
    video = pad_tensor(video, pad_left, pad_right)
    keypoint = pad_tensor(keypoint, pad_left, pad_right)
    
    #make slided inputs
    video_s = torch.zeros(num_clips, win_size, video.shape[1], video.shape[2], video.shape[3]).to(video.device)  #N,W,C,H,W
    keypoint_s = torch.zeros(num_clips, win_size, keypoint.shape[1], keypoint.shape[2]).to(video.device)  #N,W,K,3
    for i in range(num_clips):
        st = i * stride
        video_s[i, ...] = video[st:st+win_size, ...]
        keypoint_s[i, ...] = keypoint[st:st+win_size, ...]
    
    del video; del keypoint
    return video_s, keypoint_s


def evaluation_slide(model, cslr_dataloader, cfg, 
        tb_writer=None, wandb_run=None,
        epoch=None, global_step=None,
        generate_cfg={}, save_dir=None, 
        return_prob=False, return_others=False,
        model_ex=None, split='test', save_fea=False, pred_src='ensemble'):
    logger = get_logger()
    logger.info(generate_cfg)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    print()
    with open(cfg['data']['vocab_file'], 'rb') as f:
        vocab = json.load(f)
    dataset_name = cfg['data']['dataset_name']

    cls_num = len(vocab)
    if '<blank>' in vocab:
        ctc_decoder_vocab = [chr(x) for x in range(20000, 20000+cls_num)]
        blank_id = vocab.index('<blank>')
    else:
        ctc_decoder_vocab = [chr(x) for x in range(20000, 20000+cls_num+1)]
        blank_id = cls_num
        vocab.append('<blank>')
    
    head_split_setting = cfg['model']['RecognitionNetwork']['visual_head'].get('split_setting', None)
    use_bag_fc = (head_split_setting is not None and 'bag_fc' in head_split_setting)

    if is_main_process() and os.environ.get('enable_pbar', '1') == '1':
        pbar = ProgressBar(n_total=len(cslr_dataloader), desc=cslr_dataloader.dataset.split.upper())
    else:
        pbar = None
    if epoch != None:
        logger.info('------------------Evaluation epoch={} {} examples #={}---------------------'.format(epoch, cslr_dataloader.dataset.split, len(cslr_dataloader.dataset)))
    elif global_step != None:
        logger.info('------------------Evaluation global step={} {} examples #={}------------------'.format(global_step, cslr_dataloader.dataset.split, len(cslr_dataloader.dataset)))
    
    model.eval()
    if model_ex:
        model_ex.eval()
    val_stat = defaultdict(float)
    name_prob = {}
    # decode_method_lst = ['naive_greedy']
    win_size = cfg['data'].get('win_size', 16)
    stride = cfg['data'].get('stride', 1)
    split_size = cfg['data'].get('split_size', 8)
    thr_lst = cfg['data'].get('prob_thr', [0.2, -1])
    blank_thr = cfg['data'].get('blank_thr', 0.5)
    decode_method_lst = ['window_greedy_3', 'window_greedy_5', 'window_greedy_7', 'window_greedy_9', 'window_greedy_11', 'window_greedy_13', 'naive_greedy']

    if save_fea or use_bag_fc or model_ex:
        feas_rgb = []
        def save_feas_rgb(module, input, output):
            feas_rgb.append(output)
        layer_rgb = model.recognition_network.visual_backbone_twostream.rgb_stream.backbone.base[-4]  #-4,-1
        layer_rgb.register_forward_hook(save_feas_rgb)
        feas_kp = []
        def save_feas_kp(module, input, output):
            feas_kp.append(output)
        layer_kp = model.recognition_network.visual_backbone_twostream.pose_stream.backbone.base[-4]
        layer_kp.register_forward_hook(save_feas_kp)

        feas_rgb_blk5 = []
        def save_feas_rgb_blk5(module, input, output):
            feas_rgb_blk5.append(output)
        layer_rgb_blk5 = model.recognition_network.visual_backbone_twostream.rgb_stream.backbone.base[-1]  #-4,-1
        layer_rgb_blk5.register_forward_hook(save_feas_rgb_blk5)
        feas_kp_blk5 = []
        def save_feas_kp_blk5(module, input, output):
            feas_kp_blk5.append(output)
        layer_kp_blk5 = model.recognition_network.visual_backbone_twostream.pose_stream.backbone.base[-1]
        layer_kp_blk5.register_forward_hook(save_feas_kp_blk5)

    save_step = 100

    with torch.no_grad():
        if use_bag_fc:
            logger.info('use bag fc')
            bag_fc_rgb = model.recognition_network.visual_head.bag_fc
            bag_fc_kp = model.recognition_network.visual_head_keypoint.bag_fc
            bag_fc_fuse = model.recognition_network.visual_head_fuse.bag_fc

        for thr in thr_lst:
            results = defaultdict(dict)
            feas = {'rgb': {}, 'keypoint': {}, 'rgb_blk5': {}, 'keypoint_blk5': {}}
            #load exist
            fea_fname = os.path.join(save_dir, '{}_features.pkl'.format(split))
            if save_fea and os.path.exists(fea_fname):
                with open(fea_fname, 'rb') as f:
                    exist = pickle.load(f)
                feas['rgb'].update(exist['rgb'])
                feas['keypoint'].update(exist['keypoint'])
                feas['rgb_blk5'].update(exist['rgb_blk5'])
                feas['keypoint_blk5'].update(exist['keypoint_blk5'])

            logits_dict = {}
            logger.info(f'window size: {win_size}, stride: {stride}, prob_thr: {thr}, blank_thr: {blank_thr}')
            for step, batch in enumerate(cslr_dataloader):
                #forward -- loss
                name = batch['names'][0]
                if save_fea and name in feas['rgb']:
                    continue
                # batch = move_to_device(batch, cfg['device'])
                video_s, keypoint_s = sliding_windows(batch['sgn_videos'][0], batch['sgn_keypoints'][0], win_size=win_size, stride=stride, save_fea=save_fea)
                video_s_lst, keypoint_s_lst = video_s.split(split_size, dim=0), keypoint_s.split(split_size, dim=0)  #S,W,C,H,W
                
                final_decode_op = []
                all_decode_op = []
                final_gls_logits = []
                all_gls_logits = []
                feas_rgb, feas_kp, feas_rgb_blk5, feas_kp_blk5 = [], [], [], []
                for v_s, k_s in zip(video_s_lst, keypoint_s_lst):
                    v_s, k_s, batch['labels'] = v_s.to(cfg['device']), k_s.to(cfg['device']), batch['labels'].to(cfg['device'])
                    if len(cfg['data']['input_streams']) == 2:
                        sgn_videos = [v_s]
                        sgn_keypoints = [k_s]
                    elif len(cfg['data']['input_streams']) == 4:
                        sgn_videos = [v_s]
                        sgn_videos.append(sgn_videos[-1][:, win_size//4:win_size//4+win_size//2, ...].contiguous())
                        sgn_keypoints = [k_s]
                        sgn_keypoints.append(sgn_keypoints[-1][:, win_size//4:win_size//4+win_size//2, ...].contiguous())

                    forward_output = model(is_train=False, labels=batch['labels'], sgn_videos=sgn_videos, sgn_keypoints=sgn_keypoints, epoch=epoch)
                    
                    #rgb/keypoint/fuse/ensemble_last_logits
                    if pred_src == 'ensemble':
                        gls_logits = forward_output['ensemble_last_gloss_logits']
                    elif pred_src == 'fuse':
                        gls_logits = forward_output['fuse_gloss_logits']
                    
                    # if mask_entry is not None:
                    #     gls_logits[:, mask_entry] = -99999

                    gls_prob = gls_logits.softmax(dim=-1)
                    max_prob = gls_prob.amax(dim=-1)
                    decode_output = model.predict_gloss_from_logits(gloss_logits=gls_logits, k=10)  #[B,N]
                    if thr <= 0.2 or torch.sum(max_prob > thr).item() > 0:
                    # if torch.sum(max_prob > thr).item() > 0:
                        # decode_output[max_prob <= thr] = 0
                        # final_decode_op.append(decode_output)
                        final_decode_op.append(decode_output[max_prob > thr])
                        final_gls_logits.append(gls_logits[max_prob > thr])
                    # else:
                    all_decode_op.append(decode_output)
                    all_gls_logits.append(gls_logits)

                final_decode_op = torch.cat(final_decode_op, dim=0)
                final_gls_logits = torch.cat(final_gls_logits, dim=0)
                # print(final_decode_op.shape)
                if final_decode_op.shape[0] == 0:
                    final_decode_op = torch.cat(all_decode_op, dim=0)
                    final_gls_logits = torch.cat(all_gls_logits, dim=0)
                # print('2nd ', final_decode_op.shape)
                logits_dict[name] = final_gls_logits.detach().cpu().numpy()

                gls_ref = batch['gls_ref'][0]

                results[name]['gls_ref'] = gls_ref
                
                #TODO: greedy; beam search w/o blank; drop window by thresholding
                if 'naive_greedy' in decode_method_lst:
                    index = final_decode_op[:, 0]
                    index = index.detach().cpu().numpy()

                    #remove repeats
                    filtered_index = []
                    for i in index:
                        if len(filtered_index) == 0:
                            filtered_index.append(i)
                        else:
                            if i != filtered_index[-1]:
                                filtered_index.append(i)
                    
                    #remove blank, although training samples may not have blank
                    filtered_index_wo_blank = []
                    for i in filtered_index:
                        if i != blank_id:
                            filtered_index_wo_blank.append(i)
                    filtered_index = filtered_index_wo_blank

                    gls_hyp = index2token(filtered_index, vocab, dataset_name)
                    # print(index, filtered_index)
                    # print(gls_hyp)
                    results[name]['naive_greedy_gls_hyp'] = ' '.join(gls_hyp)
                
                if save_fea or use_bag_fc or model_ex:
                    feas_rgb = torch.cat(feas_rgb, dim=0)  #real_T,C,T,H,W
                    print('rgb: ', feas_rgb.shape)
                    feas_rgb = F.avg_pool3d(feas_rgb, (2, feas_rgb.size(3), feas_rgb.size(4)), stride=1)
                    feas_rgb = feas_rgb.view(feas_rgb.size(0), feas_rgb.size(1), feas_rgb.size(2)).mean(dim=-1)  #real_T,C
                    feas_kp = torch.cat(feas_kp, dim=0)  #real_T,C,T,H,W
                    print('kp: ', feas_kp.shape)
                    feas_kp = F.avg_pool3d(feas_kp, (2, feas_kp.size(3), feas_kp.size(4)), stride=1)
                    feas_kp = feas_kp.view(feas_kp.size(0), feas_kp.size(1), feas_kp.size(2)).mean(dim=-1)  #real_T,C

                    feas_rgb_blk5 = torch.cat(feas_rgb_blk5, dim=0)  #real_T,C,T,H,W
                    print('rgb_blk5: ', feas_rgb_blk5.shape)
                    feas_rgb_blk5 = F.avg_pool3d(feas_rgb_blk5, (2, feas_rgb_blk5.size(3), feas_rgb_blk5.size(4)), stride=1)
                    feas_rgb_blk5 = feas_rgb_blk5.view(feas_rgb_blk5.size(0), feas_rgb_blk5.size(1), feas_rgb_blk5.size(2)).mean(dim=-1)  #real_T,C
                    feas_kp_blk5 = torch.cat(feas_kp_blk5, dim=0)  #real_T,C,T,H,W
                    print('kp_blk5: ', feas_kp_blk5.shape)
                    feas_kp_blk5 = F.avg_pool3d(feas_kp_blk5, (2, feas_kp_blk5.size(3), feas_kp_blk5.size(4)), stride=1)
                    feas_kp_blk5 = feas_kp_blk5.view(feas_kp_blk5.size(0), feas_kp_blk5.size(1), feas_kp_blk5.size(2)).mean(dim=-1)  #real_T,C

                    feas['rgb'][name] = feas_rgb.detach().cpu()
                    feas['keypoint'][name] = feas_kp.detach().cpu()
                    feas['rgb_blk5'][name] = feas_rgb_blk5.detach().cpu()
                    feas['keypoint_blk5'][name] = feas_kp_blk5.detach().cpu()

                    if save_dir and (step+1)%save_step==0:
                        with open(fea_fname, 'wb') as f:
                            pickle.dump(feas, f)
                
                if 'window_greedy_7' in decode_method_lst:
                    filter_logits = {}
                    for decode_win_size in [3,5,7,9,11,13]:
                        index = final_decode_op[:, 0]
                        # index = torch.cat([torch.zeros(5).long().to(index.device), index], dim=0)
                        index = index.detach().cpu().numpy()

                        # pad index
                        pad_left = index[0]
                        pad = np.tile(pad_left, (decode_win_size//2))
                        index = np.concatenate([pad, index])
                        pad_right = index[-1]
                        pad = np.tile(pad_right, (decode_win_size//2))
                        index = np.concatenate([index, pad])

                        # pad logits
                        final_gls_logits = pad_tensor(final_gls_logits, decode_win_size//2, decode_win_size//2)
                        
                        if use_bag_fc or model_ex:
                            rgb_feas = pad_tensor(feas['rgb_blk5'][name], decode_win_size//2, decode_win_size//2).to(model.device)
                            kp_feas = pad_tensor(feas['keypoint_blk5'][name], decode_win_size//2, decode_win_size//2).to(model.device)

                        #------------------voting------------------------------
                        win_index = []
                        win_bag_index = []
                        win_logits = []
                        fea_buffer = []
                        for st in range(index.shape[0] - decode_win_size + 1):
                            win = index[st:st+decode_win_size]
                            if use_bag_fc or model_ex:
                                if use_bag_fc:
                                    cur_rgb_feas = rgb_feas[st:st+decode_win_size].contiguous().mean(dim=0, keepdim=True)  #1,C
                                    cur_pose_feas = kp_feas[st:st+decode_win_size].contiguous().mean(dim=0, keepdim=True)  #1,C
                                    cur_fuse_feas = torch.cat([cur_rgb_feas, cur_pose_feas], dim=-1)
                                    cur_rgb_prob = bag_fc_rgb(cur_rgb_feas).softmax(dim=-1).squeeze()  #N
                                    cur_kp_prob = bag_fc_kp(cur_pose_feas).softmax(dim=-1).squeeze()  #N
                                    cur_fuse_prob = bag_fc_fuse(cur_fuse_feas).softmax(dim=-1).squeeze()  #N
                                    prob = (cur_rgb_prob + cur_kp_prob + cur_fuse_prob) / 3
                                elif model_ex:
                                    cur_rgb_feas = rgb_feas[st:st+decode_win_size].contiguous()  #w,C
                                    cur_pose_feas = kp_feas[st:st+decode_win_size].contiguous()  #w,C
                                    cur_fuse_feas = torch.cat([cur_rgb_feas, cur_pose_feas], dim=-1).unsqueeze(0)
                                    # fea_buffer.append(cur_fuse_feas)
                                    prob = None
                                    # if len(fea_buffer) == 3:
                                    # input_fea = torch.cat(fea_buffer, dim=0)
                                    logits = model_ex(is_train=False, labels=batch['labels'], sgn_videos=sgn_videos, sgn_keypoints=sgn_keypoints, epoch=epoch,
                                                    denoise_inputs={'features': cur_fuse_feas, 'labels': None})['logits']
                                    prob = logits.softmax(dim=-1).squeeze(0)
                                    # prob = logits.softmax(dim=-1)[1]
                                    # fea_buffer = []

                            else:
                                logits = final_gls_logits[st:st+decode_win_size]
                                prob = logits.softmax(dim=-1).mean(dim=0)

                            if prob is not None:
                                win_bag_index.append(torch.argmax(prob).item())

                            uniq, count = np.unique(win, return_counts=True)
                            try:
                                keep = uniq[count > decode_win_size//2]
                                win_index.append(keep.item())
                            except:
                                win_index.append(0)  #blank
                        
                        #-------------------------remove repeat-------------------
                        filtered_index = []
                        for i in win_index:
                            if len(filtered_index) == 0:
                                filtered_index.append(i)
                            else:
                                if i != filtered_index[-1]:
                                    filtered_index.append(i)

                        filtered_bag_index = []
                        for i in win_bag_index:
                            if len(filtered_bag_index) == 0:
                                filtered_bag_index.append(i)
                            else:
                                if i != filtered_bag_index[-1]:
                                    filtered_bag_index.append(i)
                        
                        #--------------------------remove blank---------------------
                        filtered_index_wo_blank = []
                        for i in filtered_index:
                            if i != blank_id:
                                filtered_index_wo_blank.append(i)
                        filtered_index = filtered_index_wo_blank

                        filtered_bag_index_wo_blank = []
                        for i in filtered_bag_index:
                            if i != blank_id:
                                filtered_bag_index_wo_blank.append(i)
                        filtered_bag_index = filtered_bag_index_wo_blank
                        # else:
                        #     filtered_index = list(index)
                        #     win_index = list(index)

                        gls_hyp = index2token(filtered_index, vocab, dataset_name)
                        raw_gls_hyp = index2token(list(index), vocab, dataset_name)
                        win_gls_hyp = index2token(win_index, vocab, dataset_name)
                        gls_bag_hyp = index2token(filtered_bag_index, vocab, dataset_name)
                        # print(index, filtered_index)
                        # print(gls_hyp)
                        # print(gls_ref)
                        results[name]['window_greedy_{}_gls_hyp'.format(decode_win_size)] = ' '.join(gls_hyp)
                        results[name]['window_greedy_{}_raw_gls_hyp'.format(decode_win_size)] = ' '.join(raw_gls_hyp)
                        results[name]['window_greedy_{}_vote_gls_hyp'.format(decode_win_size)] = ' '.join(win_gls_hyp)
                        results[name]['window_greedy_{}_gls_bag_hyp'.format(decode_win_size)] = ' '.join(gls_bag_hyp)

                if pbar:
                    pbar(step)

                # if step == 3:
                #     break
            print()
        
            evaluation_results = {}
            # decode_method_lst.remove('beam_search')
            logger.info(f'gls_prob_thr: {thr}')
            for m in decode_method_lst:
                gls_hyp = []
                gls_bag_hyp = []
                gls_ref = []
                if dataset_name in ['phoenix']:
                    clean_func = clean_phoenix_2014_trans
                else:
                    clean_func = clean_phoenix_2014
                for name, res in results.items():
                    if 'phoenix' in dataset_name:
                        gls_hyp.append(clean_func(res['{}_gls_hyp'.format(m)]))
                        # gls_bag_hyp.append(clean_func(res['{}_gls_bag_hyp'.format(m)]))
                        gls_ref.append(clean_func(res['gls_ref']))
                    else:
                        gls_hyp.append(res['{}_gls_hyp'.format(m)])
                        gls_ref.append(res['gls_ref'])
                    
                # print(gls_ref, res['gls_ref'])
                wer_results = wer_list(references=gls_ref, hypotheses=gls_hyp)
                logger.info('Decoding method: {}, WER: {:.2f}, DEL: {:.2f}, INS: {:.2f}, SUB: {:.2f}'\
                            .format(m, wer_results['wer'], wer_results['del'], wer_results['ins'], wer_results['sub']))
                evaluation_results['wer_'+m] = wer_results

            #save
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, '{}_results.pkl'.format(split)), 'wb') as f:
                    pickle.dump(results, f)
                with open(os.path.join(save_dir, '{}_evaluation_results.pkl'.format(split)), 'wb') as f:
                    pickle.dump(evaluation_results, f)
                with open(os.path.join(save_dir, '{}_logits.pkl'.format(split)), 'wb') as f:
                    pickle.dump(logits_dict, f)
                if save_fea:
                    with open(fea_fname, 'wb') as f:
                        pickle.dump(feas, f)

    logger.info('-------------------------Evaluation Finished-------------------------'.format(global_step, len(cslr_dataloader.dataset)))
    return results, evaluation_results


def eval_g2g(model, val_dataloader, cfg, 
        tb_writer=None, wandb_run=None,
        epoch=None, global_step=None,
        generate_cfg={}, save_dir=None, 
        return_prob=False, return_others=False):
    logger = get_logger()
    logger.info(generate_cfg)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    print()
    with open(cfg['data']['vocab_file'], 'rb') as f:
        vocab = json.load(f)
    dataset_name = cfg['data']['dataset_name']
    cls_num = len(vocab)

    if is_main_process() and os.environ.get('enable_pbar', '1') == '1':
        pbar = ProgressBar(n_total=len(val_dataloader), desc=val_dataloader.dataset.split.upper())
    else:
        pbar = None
    if epoch != None:
        logger.info('------------------Evaluation epoch={} {} examples #={}---------------------'.format(epoch, val_dataloader.dataset.split, len(val_dataloader.dataset)))
    elif global_step != None:
        logger.info('------------------Evaluation global step={} {} examples #={}------------------'.format(global_step, val_dataloader.dataset.split, len(val_dataloader.dataset)))
    
    model.eval()
    total_val_loss = defaultdict(float)
    with torch.no_grad():
        results = defaultdict(dict)
        for step, batch in enumerate(val_dataloader):
            #forward -- loss
            # if step == 10:
            #     break
            names = batch['names']
            batch = move_to_device(batch, cfg['device'])
            forward_output = model(is_train=False, labels=batch['labels'], sgn_videos=batch['sgn_videos'], sgn_keypoints=batch['sgn_keypoints'], epoch=epoch,
                                   translation_inputs=batch.get('translation_inputs', {}))
            for k,v in forward_output.items():
                if '_loss' in k:
                    total_val_loss[k] += v.item()
        
            generate_output = model.generate_txt(
                    transformer_inputs=forward_output['transformer_inputs'],
                    generate_cfg=generate_cfg['translation'])
            #decoded_sequences
            for name, gls_hyp, gls_ref in zip(batch['names'], generate_output['decoded_sequences'], batch['gls_ref']):
                new_gls_hyp = []
                for g in gls_hyp.split(' '):
                    if g in model.tokenizer.special_tokens:
                        continue
                    g_ = map_phoenix_gls(g)
                    new_gls_hyp.append(g_)
                results[name]['gls_hyp'], results[name]['gls_ref'] = ' '.join(new_gls_hyp), gls_ref
                # print('hyp: ', results[name]['gls_hyp'])
                # print('ref: ', results[name]['gls_ref'])

            if pbar:
                pbar(step)
        print()
        
        gls_hyp = []
        gls_ref = []
        if dataset_name in ['phoenix']:
            clean_func = clean_phoenix_2014_trans
        else:
            clean_func = clean_phoenix_2014
        for name, res in results.items():
            if 'phoenix' in dataset_name:
                gls_hyp.append(clean_func(res['gls_hyp']))
                gls_ref.append(clean_func(res['gls_ref']))
            else:
                gls_hyp.append(res['gls_hyp'])
                gls_ref.append(res['gls_ref'])
        wer_results = wer_list(references=gls_ref, hypotheses=gls_hyp)
        logger.info('WER: {:.2f}, DEL: {:.2f}, INS: {:.2f}, SUB: {:.2f}'\
                    .format(wer_results['wer'], wer_results['del'], wer_results['ins'], wer_results['sub']))

        #save
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(results, f)
            with open(os.path.join(save_dir, 'evaluation_results.pkl'), 'wb') as f:
                pickle.dump(wer_results, f)

    logger.info('-------------------------Evaluation Finished-------------------------'.format(global_step, len(val_dataloader.dataset)))
    return results, wer_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument("--config", default="configs/default.yaml", type=str, help="Training configuration file (yaml).")
    parser.add_argument("--config_ex", default=None, type=str, help="Extra config")
    parser.add_argument("--save_subdir", default='prediction_slide', type=str)
    parser.add_argument('--ckpt_name', default='best.ckpt', type=str)
    parser.add_argument('--eval_setting', default='origin', type=str)
    parser.add_argument('--blank_thr', default=0.5, type=float)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--save_fea', default=0, choices=[0,1], type=int)
    parser.add_argument('--pred_src', default='ensemble', choices=['ensemble', 'fuse'], type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg['data']['blank_thr'] = args.blank_thr
    # cfg['local_rank'], cfg['world_size'], cfg['device'] = init_DDP()
    cfg['device'] = torch.device('cuda')
    set_seed(seed=cfg["training"].get("random_seed", 42))
    model_dir = cfg['training']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    global logger
    logger = make_logger(model_dir=model_dir, log_file='prediction_slide_{}.log'.format(args.eval_setting))
    os.system('cp prediction_slide.py {}/'.format(model_dir))

    cfg_ex = None
    if args.config_ex:
        cfg_ex = load_config(args.config_ex)
        cfg_ex['device'] = cfg['device']

    dataset = build_dataset(cfg['data'], 'train')
    vocab = dataset.vocab
    cls_num = len(vocab)

    del vocab; del dataset
    model = build_model(cfg, cls_num, word_emb_tab=None)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #load model
    load_model_path = os.path.join(model_dir, 'ckpts', args.ckpt_name)
    if os.path.isfile(load_model_path):
        state_dict = torch.load(load_model_path, map_location='cuda')
        neq_load_customized(model, state_dict['model_state'], verbose=True)
        epoch, global_step = state_dict.get('epoch', 0), state_dict.get('global_step', 0)
        logger.info('Load model ckpt from ' + load_model_path)
    else:
        logger.info(f'{load_model_path} does not exist')
        epoch, global_step = 0, 0
    
    # model = DDP(model, 
    #         device_ids=[cfg['local_rank']], 
    #         output_device=cfg['local_rank'],
    #         find_unused_parameters=True)

    model_ex = None
    if cfg_ex:
        model_dir_ex = cfg_ex['training']['model_dir']
        if 'bin' in model_dir_ex:
            #fore/background model
            cls_num_ex = 2
        else:
            cls_num_ex = cls_num
        model_ex = build_model(cfg_ex, cls_num_ex, word_emb_tab=None)
        # model_ex = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_ex) 
        #load model
        load_model_path = os.path.join(model_dir_ex, 'ckpts', args.ckpt_name)
        state_dict = torch.load(load_model_path, map_location='cuda')
        neq_load_customized(model_ex, state_dict['model_state'], verbose=True)
        epoch, global_step = state_dict.get('epoch', 0), state_dict.get('global_step', 0)
        logger.info('Load model_ex ckpt from ' + load_model_path)
        # model_ex = DDP(model_ex, 
        #     device_ids=[cfg_ex['local_rank']], 
        #     output_device=cfg_ex['local_rank'],
        #     find_unused_parameters=True)

    for split in [args.split]:
        if 'train_' in split:
            split = 'train'
        logger.info('Evaluate on {} set'.format(split))

        g2g_tokenizer = model.tokenizer if cfg['task'] == 'G2G' else None
        dataloader, sampler = build_dataloader(cfg, split, task=cfg['task'], g2g_tokenizer=g2g_tokenizer, is_train=False, val_distributed=False)

        if cfg['task'] == 'ISLR':
            _, _ = evaluation_slide(model=model, cslr_dataloader=dataloader, cfg=cfg, 
                    epoch=epoch, global_step=global_step, 
                    generate_cfg=cfg['testing']['cfg'],
                    save_dir=os.path.join(model_dir,args.save_subdir,split), 
                    return_prob=True, return_others=False, model_ex=model_ex, split=args.split, save_fea=bool(args.save_fea), pred_src=args.pred_src)
        
