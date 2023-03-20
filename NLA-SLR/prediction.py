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
import time
import queue
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
from utils.metrics import compute_accuracy
from dataset.Dataloader import build_dataloader
from dataset.Dataset import build_dataset
from utils.progressbar import ProgressBar
from copy import deepcopy


def evaluation(model, val_dataloader, cfg, 
        tb_writer=None, wandb_run=None,
        epoch=None, global_step=None,
        generate_cfg={}, save_dir=None, return_prob=False):  #to-do output_dir
    logger = get_logger()
    logger.info(generate_cfg)
    print()
    vocab = val_dataloader.dataset.vocab
    split = val_dataloader.dataset.split
    cls_num = len(vocab)

    word_emb_tab = []
    if val_dataloader.dataset.word_emb_tab is not None:
        for w in vocab:
            word_emb_tab.append(torch.from_numpy(val_dataloader.dataset.word_emb_tab[w]))
        word_emb_tab = torch.stack(word_emb_tab, dim=0).float().to(cfg['device'])
    else:
        word_emb_tab = None

    if is_main_process() and os.environ.get('enable_pbar', '1') == '1':
        pbar = ProgressBar(n_total=len(val_dataloader), desc=val_dataloader.dataset.split.upper())
    else:
        pbar = None
    if epoch != None:
        logger.info('------------------Evaluation epoch={} {} examples #={}---------------------'.format(epoch, val_dataloader.dataset.split, len(val_dataloader.dataset)))
    elif global_step != None:
        logger.info('------------------Evaluation global step={} {} examples #={}------------------'.format(global_step, val_dataloader.dataset.split, len(val_dataloader.dataset)))
    
    model.eval()
    val_stat = defaultdict(float)
    results = defaultdict(dict)
    name_prob = {}
    contras_setting = cfg['model']['RecognitionNetwork']['visual_head']['contras_setting']
    pred_src = 'gloss_logits'
    with torch.no_grad():
        logits_name_lst = []
        for step, batch in enumerate(val_dataloader):
            #forward -- loss
            batch = move_to_device(batch, cfg['device'])

            forward_output = model(is_train=False, labels=batch['labels'], sgn_videos=batch['sgn_videos'], sgn_keypoints=batch['sgn_keypoints'], epoch=epoch)
            if is_main_process():
                for k,v in forward_output.items():
                    if '_loss' in k:
                        val_stat[k] += v.item()

            #rgb/keypoint/fuse/ensemble_last_logits
            for k, gls_logits in forward_output.items():
                if pred_src not in k or gls_logits == None:
                    continue
                
                logits_name = k.replace(pred_src,'')
                if 'word_fused' in logits_name:
                    continue
                logits_name_lst.append(logits_name)

                decode_output = model.predict_gloss_from_logits(gloss_logits=gls_logits, k=10)
                if return_prob:
                    gls_prob = forward_output[f'{logits_name}gloss_logits'].softmax(dim=-1)
                    if (len(cfg['data']['input_streams']) == 1 and logits_name == '') or \
                        (len(cfg['data']['input_streams']) > 1 and logits_name in ['ensemble_last_', 'fuse_', 'ensemble_all_']):
                        for i in range(gls_prob.shape[0]):
                            name = logits_name + batch['names'][i]
                            name_prob[name] = gls_prob[i]
                    gls_prob = torch.sort(gls_prob, dim=-1, descending=True)[0]
                    gls_prob = gls_prob[..., :10]  #[B,10]

                for i in range(decode_output.shape[0]):
                    name = batch['names'][i]
                    hyp = [d.item() for d in decode_output[i]]
                    results[name][f'{logits_name}hyp'] = hyp

                    if return_prob:
                        prob = [d.item() for d in gls_prob[i]]
                        results[name][f'{logits_name}prob'] = prob

                    ref = batch['labels'][i].item()
                    results[name]['ref'] = ref

            if pbar:
                pbar(step)
        print()
    
    #logging and tb_writer
    if is_main_process():
        for k, v in val_stat.items():
            if '_loss' in k:
                logger.info('{} Average:{:.2f}'.format(k, v/len(val_dataloader)))
            if wandb_run:
                wandb.log({f'eval/{k}': v/len(val_dataloader)})
    
    per_ins_stat_dict, per_cls_stat_dict = compute_accuracy(results, logits_name_lst, cls_num, cfg['device'])

    #save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'results_{}.pkl'.format(cfg['local_rank'])), 'wb') as f:
            pickle.dump(results, f)

    if return_prob:
        with open(os.path.join(save_dir, 'name_prob.pkl'), 'wb') as f:
            pickle.dump(name_prob, f)
    logger.info('-------------------------Evaluation Finished-------------------------'.format(global_step, len(val_dataloader.dataset)))
    return per_ins_stat_dict, per_cls_stat_dict, results, name_prob


def sync_results(per_ins_stat_dict, per_cls_stat_dict, save_dir=None, wandb_run=None, sync=True):
    logger = get_logger()

    if sync:
        for d in [per_ins_stat_dict, per_cls_stat_dict]:
            for k,v in d.items():
                synchronize()
                torch.distributed.all_reduce(v)

    evaluation_results = {}
    if is_main_process():
        for k, per_ins_stat in per_ins_stat_dict.items():
            correct, correct_5, correct_10, num_samples = per_ins_stat
            logger.info('#samples: {}'.format(num_samples))
            evaluation_results[f'{k}per_ins_top_1'] = (correct / num_samples).item()
            logger.info('-------------------------{}Per-instance ACC Top-1: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_ins_top_1']))
            evaluation_results[f'{k}per_ins_top_5'] = (correct_5 / num_samples).item()
            logger.info('-------------------------{}Per-instance ACC Top-5: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_ins_top_5']))
            evaluation_results[f'{k}per_ins_top_10'] = (correct_10 / num_samples).item()
            logger.info('-------------------------{}Per-instance ACC Top-10: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_ins_top_10']))

        # one class missing in the test set of WLASL_2000
        for k, per_cls_stat in per_cls_stat_dict.items():
            top1_t, top1_f, top5_t, top5_f, top10_t, top10_f = per_cls_stat
            evaluation_results[f'{k}per_cls_top_1'] = np.nanmean((top1_t / (top1_t+top1_f)).cpu().numpy())
            logger.info('-------------------------{}Per-class ACC Top-1: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_cls_top_1']))
            evaluation_results[f'{k}per_cls_top_5'] = np.nanmean((top5_t / (top5_t+top5_f)).cpu().numpy())
            logger.info('-------------------------{}Per-class ACC Top-5: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_cls_top_5']))
            evaluation_results[f'{k}per_cls_top_10'] = np.nanmean((top10_t / (top10_t+top10_f)).cpu().numpy())
            logger.info('-------------------------{}Per-class ACC Top-10: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_cls_top_10']))

        if wandb_run:
            for k, v in evaluation_results.items():
                wandb.log({f'eval/{k}': 100*v})

        if save_dir:
            with open(os.path.join(save_dir, 'evaluation_results.pkl'), 'wb') as f:
                pickle.dump(evaluation_results, f)
    if sync:
        synchronize()
    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument("--config", default="configs/default.yaml", type=str, help="Training configuration file (yaml).")
    parser.add_argument("--save_subdir", default='prediction', type=str)
    parser.add_argument('--ckpt_name', default='best.ckpt', type=str)
    parser.add_argument('--eval_setting', default='origin', type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg['local_rank'], cfg['world_size'], cfg['device'] = init_DDP()
    set_seed(seed=cfg["training"].get("random_seed", 42))
    model_dir = cfg['training']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    global logger
    logger = make_logger(model_dir=model_dir, log_file='prediction_{}_{}.log'.format(args.eval_setting, cfg['local_rank']))

    dataset = build_dataset(cfg['data'], 'train')
    vocab = dataset.vocab
    cls_num = len(vocab)
    word_emb_tab = []
    if dataset.word_emb_tab is not None:
        for w in vocab:
            word_emb_tab.append(torch.from_numpy(dataset.word_emb_tab[w]))
        word_emb_tab = torch.stack(word_emb_tab, dim=0).float().to(cfg['device'])
    else:
        word_emb_tab = None
    del vocab; del dataset
    model = build_model(cfg, cls_num, word_emb_tab=word_emb_tab)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    #load model
    load_model_path = os.path.join(model_dir,'ckpts',args.ckpt_name)
    if os.path.isfile(load_model_path):
        state_dict = torch.load(load_model_path, map_location='cuda')
        neq_load_customized(model, state_dict['model_state'], verbose=True)
        epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
        logger.info('Load model ckpt from '+load_model_path)
    else:
        logger.info(f'{load_model_path} does not exist')
        epoch, global_step = 0, 0
    
    model = DDP(model, 
            device_ids=[cfg['local_rank']], 
            output_device=cfg['local_rank'],
            find_unused_parameters=True)

    for split in ['test']:
        logger.info('Evaluate on {} set'.format(split))
        if args.eval_setting == 'origin':
            dataloader, sampler = build_dataloader(cfg, split, is_train=False, val_distributed=True)
            per_ins_stat, per_cls_stat, _, _ = evaluation(model=model.module, val_dataloader=dataloader, cfg=cfg, 
                    epoch=epoch, global_step=global_step, 
                    generate_cfg=cfg['testing']['cfg'],
                    save_dir=os.path.join(model_dir,args.save_subdir,split), return_prob=True)
            
            sync_results(per_ins_stat, per_cls_stat)
        
        elif args.eval_setting == '3x_pad':
            test_p = ['start', 'end', 'central']
            test_m = ['start_pad', 'end_pad', 'pad']
            all_prob = {}
            for t_p, t_m in zip(test_p, test_m):
                logger.info('----------------------------------crop position: {}----------------------------'.format(t_p))
                new_cfg = deepcopy(cfg)
                new_cfg['data']['transform_cfg']['index_setting'][2] = t_p
                new_cfg['data']['transform_cfg']['index_setting'][3] = t_m
                dataloader, sampler = build_dataloader(new_cfg, split, is_train=False, val_distributed=False)
                per_ins_stat, per_cls_stat, results, name_prob = evaluation(model=model.module, val_dataloader=dataloader, cfg=new_cfg, 
                                        epoch=epoch, global_step=global_step, 
                                        generate_cfg=cfg['testing']['cfg'],
                                        save_dir=os.path.join(model_dir,args.save_subdir,split), return_prob=True)
                all_prob[t_p] = name_prob
            with open(os.path.join(model_dir,args.save_subdir, split, 'prob_3x_pad.pkl'), 'wb') as f:
                pickle.dump(all_prob, f)
            
            if len(cfg['data']['input_streams']) == 1:
                logits_name_lst = ['']
            elif len(cfg['data']['input_streams']) == 4:
                logits_name_lst = ['ensemble_last_', 'ensemble_all_']
            else:
                logits_name_lst = ['ensemble_last_', 'fuse_']
            
            evaluation_results = compute_accuracy(results, logits_name_lst, cls_num, cfg['device'], 
                all_prob=all_prob, eval_setting=args.eval_setting)
            for logits_name in logits_name_lst:
                logger.info('-------------------------{}Per-instance ACC Top-1: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_1']))
                logger.info('-------------------------{}Per-instance ACC Top-5: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_5']))
                logger.info('-------------------------{}Per-instance ACC Top-10: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_10']))

                # one class missing in the test set of WLASL_2000
                logger.info('-------------------------{}Per-class ACC Top-1: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_1']))
                logger.info('-------------------------{}Per-class ACC Top-5: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_5']))
                logger.info('-------------------------{}Per-class ACC Top-10: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_10']))
                logger.info('-------------------------Evaluation Finished-------------------------')
            