from typing import Text
import warnings
from modelling.model import build_model
from utils.optimizer import build_optimizer, build_scheduler, update_moving_average
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
from torch.utils.tensorboard import SummaryWriter
from utils.misc import (
    load_config,
    log_cfg,
    load_checkpoint,
    move_to_device, neq_load_customized,
    make_model_dir,
    make_logger, make_writer, make_wandb,
    set_seed,
    symlink_update,
    is_main_process, init_DDP,
    synchronize,
    get_fbank
)
from dataset.Dataloader import build_dataloader
from dataset.Dataset import build_dataset
from utils.progressbar import ProgressBar
from prediction import evaluation, sync_results, eval_denoise
from prediction_slide import eval_g2g
import wandb
from torch.cuda.amp import autocast, GradScaler
# from apex import amp
import math
from collections import defaultdict
from copy import deepcopy


def save_model(model, optimizer, scheduler, output_file, epoch=None, global_step=None, current_score=None):
    base_dir = os.path.dirname(output_file)
    os.makedirs(base_dir, exist_ok=True)
    state = {
        'epoch': epoch,
        'global_step':global_step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_score': best_score,
        'current_score': current_score,
    }
    torch.save(state, output_file)
    logger.info('Save model state as '+ output_file)
    return output_file


def evaluate_and_save(model, optimizer, scheduler, val_dataloader, cfg, 
                        tb_writer, wandb_run=None,
                        epoch=None, global_step=None, generate_cfg={}):
    tag = 'epoch_{:02d}'.format(epoch) if epoch!=None else 'step_{}'.format(global_step)
    #save
    global best_score, ckpt_queue
    if cfg['task'] == 'ISLR':
        per_ins_stat, per_cls_stat, _, _, _ = evaluation(
            model=model, val_dataloader=val_dataloader, cfg=cfg, 
            tb_writer=tb_writer, wandb_run=wandb_run,
            epoch=epoch, global_step=global_step, generate_cfg=generate_cfg,
            save_dir=None)
            # save_dir=os.path.join(cfg['training']['model_dir'],'validation',tag))
        evaluation_results = sync_results(per_ins_stat, per_cls_stat, save_dir=None, wandb_run=wandb_run)
    elif cfg['task'] == 'G2G' and is_main_process():
        _, evaluation_results = eval_g2g(model=model, val_dataloader=val_dataloader, cfg=cfg, 
                                        tb_writer=tb_writer, wandb_run=wandb_run,
                                        epoch=epoch, global_step=global_step, generate_cfg=generate_cfg,
                                        save_dir=None)
    elif cfg['task'] == 'bag_denoise' and is_main_process():
        evaluation_results = eval_denoise(model=model, val_dataloader=val_dataloader, cfg=cfg, 
                                        tb_writer=tb_writer, wandb_run=wandb_run,
                                        epoch=epoch, global_step=global_step, generate_cfg=generate_cfg,
                                        save_dir=None)
    metric = 'wer' if cfg['task'] == 'G2G' else 'acc'

    if is_main_process():
        if metric == 'wer':
            score = evaluation_results['wer']
        else:
            if len(cfg['data']['input_streams']) == 1:
                try:
                    score = evaluation_results['per_ins_top_1']
                except:
                    score = evaluation_results['frame_ensemble_per_ins_top_1']
            else:
                score = evaluation_results['ensemble_last_per_ins_top_1']
        if metric == 'acc':
            best_score = max(best_score, score)
            logger.info('best_score={:.2f}'.format(100*best_score))
        else:
            best_score = min(best_score, score)
            logger.info('best_score={:.2f}'.format(best_score))
        
        ckpt_file = save_model(model=model, optimizer=optimizer, scheduler=scheduler,
            output_file=os.path.join(cfg['training']['model_dir'],'ckpts',tag+'.ckpt'),
            epoch=epoch, global_step=global_step,
            current_score=score)

        if best_score==score:
            os.system('cp {} {}'.format(ckpt_file, os.path.join(cfg['training']['model_dir'],'ckpts','best.ckpt')))
            logger.info('Current Best Epoch: {:d}'.format(epoch))
        if ckpt_queue.full():
            to_delete = ckpt_queue.get()
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                logger.warning(
                    "Wanted to delete old checkpoint %s but " "file does not exist.",
                    to_delete,
                )
        ckpt_queue.put(ckpt_file)
    synchronize()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    args = parser.parse_args()
    assert 'LOCAL_RANK' in os.environ, 'Only support distributed training now!'
    
    cfg = load_config(args.config)

    # PREPARATION
    cfg['local_rank'], cfg['world_size'], cfg['device'] = init_DDP()
    cfg['rank'] = torch.distributed.get_rank()
    set_seed(seed=cfg["training"].get("random_seed", 42))
    model_dir = make_model_dir(
        model_dir=cfg['training']['model_dir'], 
        overwrite=(cfg['training'].get('overwrite',False) and not cfg['training'].get('from_ckpt',False)))
    global logger
    logger = make_logger(
        model_dir=model_dir,
        log_file='train.rank{}.log'.format(cfg['rank']))
    tb_writer = None #SummaryWriter or None (local_rank>=1)
    wandb_run = make_wandb(model_dir=model_dir, cfg=cfg)
    if is_main_process():
        os.system('cp {} {}/'.format(args.config, model_dir))
        os.system('cp -r modelling/*.py {}/'.format(model_dir))
        os.system('cp -r utils/*.py {}/'.format(model_dir))
        os.system('cp -r dataset/*.py {}/'.format(model_dir))
        os.system('cp prediction.py {}/'.format(model_dir))
        os.system('cp prediction_slide.py {}/'.format(model_dir))
    synchronize()

    #MODEL
    #!!! Must-do sync_batch
    dataset = build_dataset(cfg['data'], 'train')
    vocab = dataset.vocab
    cls_num = len(vocab)
    # make word emb table as tensor
    word_emb_tab = []
    if dataset.word_emb_tab is not None:
        for w in vocab:
            word_emb_tab.append(torch.from_numpy(dataset.word_emb_tab[w]))
        word_emb_tab = torch.stack(word_emb_tab, dim=0).float().to(cfg['device'])
    else:
        word_emb_tab = None
    del dataset
    model = build_model(cfg, cls_num, word_emb_tab=word_emb_tab)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('# Total parameters = {}'.format(total_params))
    logger.info('# Total trainable parameters = {}'.format(total_params_trainable))
    
    #load pretrained ckpt
    ckpt_file = cfg['training'].get('load_ckpt', None)
    if ckpt_file:
        pretrained_dict = torch.load(ckpt_file, map_location='cpu')['model_state']
        updated_dict = {}
        if 'multi_head_fixbug_notsharedhead' in ckpt_file:
            for k,v in pretrained_dict.items():
                if 'rgb_pyramid' in k:
                    updated_dict[k.replace('rgb_pyramid', 'rgb_stream.pyramid')] = v
                elif 'pose_pyramid' in k:
                    updated_dict[k.replace('pose_pyramid', 'pose_stream.pyramid')] = v
                else:
                    updated_dict[k] = v
        elif 'rgb_s3d' in ckpt_file and cfg['data']['input_streams'] == ['keypoint']:
            for k,v in pretrained_dict.items():
                if 'visual_backbone' in k:
                    updated_dict[k.replace('visual_backbone', 'visual_backbone_keypoint')] = v
                elif 'visual_head' in k:
                    updated_dict[k.replace('visual_head', 'visual_head_keypoint')] = v
        else:
            updated_dict = pretrained_dict.copy()
        del pretrained_dict
        neq_load_customized(model, updated_dict, verbose=True)
        logger.info("Load ckpt {:s}".format(ckpt_file))
    
    use_amp = cfg['training'].get('amp', False)
    if not use_amp:
        model = DDP(model, 
            device_ids=[cfg['local_rank']], 
            output_device=cfg['local_rank'],
            find_unused_parameters=True)
    
    #DATASET
    g2g_tokenizer = model.module.tokenizer if cfg['task'] == 'G2G' else None
    train_dataloader, train_sampler = build_dataloader(cfg, 'train', task=cfg['task'], g2g_tokenizer=g2g_tokenizer, is_train=True)
    dev_dataloader, dev_sampler = build_dataloader(cfg, 'dev', task=cfg['task'], g2g_tokenizer=g2g_tokenizer, is_train=False, val_distributed=(cfg['task'] in ['ISLR']))
    update_base = cfg['data'].get('update_base', None)
    if update_base is not None:
        upd_start, upd_freq = map(int, update_base.split('_'))
        cfg_upd = deepcopy(cfg)
        cfg_upd['data']['train'] = cfg_upd['data']['base']
        cfg_upd['data']['temp_jitter'] = 2
        cfg_upd['training']['batch_size'] = 1
        upd_base_dataloader, upd_base_sampler = build_dataloader(cfg_upd, 'train', is_train=False, val_distributed=True)

    #OPTIMIZATION
    optimizer = build_optimizer(config=cfg['training']['optimization'], model=model.module if not use_amp else model) #assign different learning rates for different modules
    scheduler, scheduler_type = build_scheduler(config=cfg['training']['optimization'], optimizer=optimizer)
    assert scheduler_type == 'epoch'
    start_epoch, total_epoch, global_step = 0, cfg['training']['total_epoch'], 0
    val_unit, val_freq = cfg['training']['validation']['unit'], cfg['training']['validation']['freq']
    global ckpt_queue, best_score
    ckpt_queue = queue.Queue(maxsize=cfg['training']['keep_last_ckpts'])
    best_score = -10000 if cfg['task'] in ['ISLR', 'bag_denoise'] else 10000

    #RESUME TRAINING
    if cfg['training'].get('from_ckpt', False):
        synchronize()
        if os.path.exists(os.path.join(model_dir, 'ckpts')):
            ckpt_lst = sorted([f for f in os.listdir(os.path.join(model_dir, 'ckpts')) if 'best' not in f and 'first_update' not in f])
            if len(ckpt_lst) > 0:
                ckpt_lst = sorted(os.listdir(os.path.join(model_dir, 'ckpts')))
                latest_ckpt = ckpt_lst[-1]
                latest_ckpt = os.path.join(model_dir, 'ckpts', latest_ckpt)
                state_dict = torch.load(latest_ckpt, 'cuda:{:d}'.format(cfg['local_rank']))
                model.module.load_state_dict(state_dict['model_state'])
                optimizer.load_state_dict(state_dict['optimizer_state'])
                scheduler.load_state_dict(state_dict['scheduler_state'])
                start_epoch = state_dict['epoch']+1 if state_dict['epoch'] is not None else int(latest_ckpt.split('_')[-1][:-5])+1
                global_step = state_dict['global_step']+1 if state_dict['global_step'] is not None else 0
                best_score = state_dict['best_score']
                # change dataloader order
                torch.manual_seed(cfg["training"].get("random_seed", 42)+start_epoch)
                train_dataloader, train_sampler = build_dataloader(cfg, 'train', task=cfg['task'], g2g_tokenizer=g2g_tokenizer, is_train=True)
                dev_dataloader, dev_sampler = build_dataloader(cfg, 'dev', task=cfg['task'], g2g_tokenizer=g2g_tokenizer, is_train=False, val_distributed=(cfg['task'] in ['ISLR']))
                logger.info('Sucessfully resume training from {:s}'.format(latest_ckpt))
                print(best_score)

    #Others (pbar, tb)
    if is_main_process():
        if os.environ.get('enable_pbar', '1')=='1':
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        else:
            pbar = None
        tb_writer = None
    else:
        pbar, tb_writer = None, None
        
    # logger.info('Evaluation at the beginning ...')
    # if is_main_process():
    #     evaluate_and_save(
    #         model=model.module, optimizer=optimizer, scheduler=scheduler,
    #         val_dataloader=dev_dataloader,
    #         cfg=cfg, tb_writer=None, wandb_run=wandb_run, epoch=-1,
    #         generate_cfg=cfg['training']['validation']['cfg'])  

    #iteration
    if use_amp == 'torch':
        scaler = GradScaler()
        torch.backends.cudnn.benchmark = True

    grad_acc_step = cfg['training'].get('grad_acc_step', 1)
    contras_setting = cfg['model']['RecognitionNetwork']['visual_head']['contras_setting'] if cfg['task']=='ISLR' else None
    fbank_pose, fbank_rgb = None, None
    epoch_no = 0
    max_step = cfg['training'].get('max_step', float('inf'))
    for epoch_no in range(start_epoch, total_epoch):
        train_sampler.set_epoch(epoch_no)

        print()
        logger.info('Epoch {}, Training examples {}'.format(epoch_no, len(train_dataloader.dataset)))
        if 'warmup' not in cfg['training']['optimization']['scheduler']:
            scheduler.step()
        stat = {}
        for step, batch in enumerate(train_dataloader):
            # print(batch['names'][0])
            if ('debug' in model_dir and step==10) or (step >= max_step):
                break
            model.module.set_train()
            # print(batch['labels'])
            if use_amp == 'torch':
                with autocast():
                    output = model(is_train=True, labels=batch['labels'], sgn_videos=batch['sgn_videos'], sgn_keypoints=batch['sgn_keypoints'], epoch=epoch_no)
            else:
                output = model(is_train=True, labels=batch['labels'], sgn_videos=batch['sgn_videos'], sgn_keypoints=batch['sgn_keypoints'], epoch=epoch_no,
                                fbank_rgb=fbank_rgb, fbank_pose=fbank_pose, start_idx=batch['start_idx'], aug=batch['aug'], bag_labels=batch['bag_labels'],
                                iou_labels=batch['iou_labels'], temp_idx=batch['temp_idx'], translation_inputs=batch.get('translation_inputs', {}),
                                denoise_inputs=batch.get('denoise_inputs', {}))
                # print('inp: ', batch['translation_inputs']['input_ids'])
                # print('lab: ', batch['translation_inputs']['labels'])
            
            #optimize
            if use_amp == 'torch':
                # if grad_acc_step > 1:
                # output['total_loss'] = output['total_loss'] / grad_acc_step
                scaler.scale(output['total_loss']).backward()
                # if ((step+1) % grad_acc_step)==0:
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
            else:
                with torch.autograd.set_detect_anomaly(True):
                    # print(output['total_loss'].item(), output['recognition_loss'].item(), output['contras_loss_'].item())
                # print((batch['sgn_keypoints']==0).sum())
                # if not torch.isnan(output['total_loss']):
                    output['total_loss'].backward()
                # else:
                    # logger.info('nan')
                
                optimizer.step()
                model.zero_grad()

            if contras_setting is not None and 'ema' in contras_setting:
                beta_init = cfg['training']['optimization'].get('beta_ema', 0.99)
                beta = 1.0 - (1.0-beta_init)*(math.cos(math.pi*epoch_no/total_epoch)+1.0)/2.0
                ma_model_lst, cur_model_lst, extra_model_lst = [], [], []
                if cfg['data']['input_streams'] == ['rgb']:
                    ma_model_lst.append(model.module.recognition_network.visual_head.gloss_output_layer)
                    cur_model_lst.append(model.module.recognition_network.visual_head.word_fused_gloss_output_layer)
                elif cfg['data']['input_streams'] == ['keypoint']:
                    ma_model_lst.append(model.module.recognition_network.visual_head_keypoint.gloss_output_layer)
                    cur_model_lst.append(model.module.recognition_network.visual_head_keypoint.word_fused_gloss_output_layer)
                elif len(cfg['data']['input_streams']) == 2:
                    language_apply_to = cfg['model']['RecognitionNetwork'].get('language_apply_to', 'rgb_keypoint_joint_traj')
                    if 'rgb' in language_apply_to:
                        ma_model_lst.append(model.module.recognition_network.visual_head.gloss_output_layer)
                        cur_model_lst.append(model.module.recognition_network.visual_head.word_fused_gloss_output_layer)
                        if 'xmodal' in contras_setting:
                            extra_model_lst.append(model.module.recognition_network.visual_head.xmodal_fused_gloss_output_layer)
                    if 'keypoint' in language_apply_to:
                        ma_model_lst.append(model.module.recognition_network.visual_head_keypoint.gloss_output_layer)
                        cur_model_lst.append(model.module.recognition_network.visual_head_keypoint.word_fused_gloss_output_layer)
                        if 'xmodal' in contras_setting:
                            extra_model_lst.append(model.module.recognition_network.visual_head_keypoint.xmodal_fused_gloss_output_layer)
                    if 'joint' in language_apply_to:
                        ma_model_lst.append(model.module.recognition_network.visual_head_fuse.gloss_output_layer)
                        cur_model_lst.append(model.module.recognition_network.visual_head_fuse.word_fused_gloss_output_layer)
                        if 'xmodal' in contras_setting:
                            # extra_model_lst.append(model.module.recognition_network.visual_head_fuse.xmodal_fused_gloss_output_layer)
                            extra_model_lst.append(None)
                    if 'traj' in language_apply_to:
                        ma_model_lst.append(model.module.recognition_network.visual_head_traj.gloss_output_layer)
                        cur_model_lst.append(model.module.recognition_network.visual_head_traj.word_fused_gloss_output_layer)
                elif len(cfg['data']['input_streams']) == 4:
                    language_apply_to = cfg['model']['RecognitionNetwork'].get('language_apply_to', 'rgb_keypoint_joint_traj')
                    for k, v in model.module.recognition_network.head_dict.items():
                        if v is not None:
                            if ('rgb' in language_apply_to and k in ['rgb-h', 'rgb-l'])\
                                or ('keypoint' in language_apply_to and k in ['kp-h', 'kp-l'])\
                                or ('joint' in language_apply_to and k in ['fuse', 'fuse-h', 'fuse-l', 'fuse-x-rgb', 'fuse-x-kp']):
                                ma_model_lst.append(v.gloss_output_layer)
                                cur_model_lst.append(v.word_fused_gloss_output_layer)
                    # print('ema')
                for m,c in zip(ma_model_lst, cur_model_lst):
                    update_moving_average(beta, m, c)
            
            for k, v in output.items():
                if '_loss' in k:
                    stat[k] = stat.get(k, 0.0) + v.item()
                # if '_weight' in k:
                #     stat[k] = stat.get(k, 0.0) + v.mean().item()

            #validation and save (last ckpt and the best ckpt)
            # if is_main_process() and val_unit=='step' and global_step%val_freq==0:
            #     evaluate_and_save(
            #         model=model.module, optimizer=optimizer, scheduler=scheduler,
            #         val_dataloader=dev_dataloader,
            #         cfg=cfg, tb_writer=tb_writer, wandb_run=wandb_run,
            #         global_step=global_step,
            #         generate_cfg=cfg['training']['validation']['cfg'])

            #misc
            global_step += 1
            if pbar:
                pbar(step)

        if 'warmup' in cfg['training']['optimization']['scheduler']:
            scheduler.step()

        synchronize()
        #visualization and logging
        if is_main_process() and wandb_run:
            lr = scheduler.optimizer.param_groups[0]["lr"]
            if wandb_run != None:
                wandb.log({k: v/len(train_dataloader) for k, v in stat.items() if '_loss' in k or '_weight' in k})
                wandb.log({'learning_rate': lr})

        if val_unit=='epoch' and epoch_no%val_freq==0:
            evaluate_and_save(
                model=model.module, optimizer=optimizer, scheduler=scheduler,
                val_dataloader=dev_dataloader,
                cfg=cfg, tb_writer=tb_writer, wandb_run=wandb_run,
                epoch=epoch_no,
                generate_cfg=cfg['training']['validation']['cfg'])
    
    evaluate_and_save(
        model=model.module, optimizer=optimizer, scheduler=scheduler,
        val_dataloader=dev_dataloader,
        cfg=cfg, tb_writer=None, wandb_run=wandb_run,
        epoch=epoch_no,
        generate_cfg=cfg['training']['validation']['cfg'])
    
    if cfg['task'] == 'G2G' and is_main_process():
        load_model_path = os.path.join(cfg['training']['model_dir'],'ckpts','best.ckpt')
        state_dict = torch.load(load_model_path, map_location='cuda')
        model.module.load_state_dict(state_dict['model_state'])

        eval_g2g(model=model.module, val_dataloader=dev_dataloader, cfg=cfg, 
                    tb_writer=tb_writer, wandb_run=wandb_run,
                    epoch=999, global_step=global_step, generate_cfg=cfg['testing']['cfg'],
                    save_dir=os.path.join(model_dir, 'dev'))
        test_dataloader, test_sampler = build_dataloader(cfg, 'test', task=cfg['task'], g2g_tokenizer=g2g_tokenizer, is_train=False, val_distributed=(cfg['task'] in ['ISLR']))
        eval_g2g(model=model.module, val_dataloader=test_dataloader, cfg=cfg, 
                    tb_writer=tb_writer, wandb_run=wandb_run,
                    epoch=999, global_step=global_step, generate_cfg=cfg['testing']['cfg'],
                    save_dir=os.path.join(model_dir, 'test'))

    #test
    #load model
    # if is_main_process():
    #     load_model_path = os.path.join(cfg['training']['model_dir'],'ckpts','best.ckpt')
    #     state_dict = torch.load(load_model_path, map_location='cuda')
    #     model.module.load_state_dict(state_dict['model_state'])
    #     epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
    #     logger.info('Load model ckpt from '+load_model_path)
    #     split = 'dev'
    #     logger.info('Evaluate on {} set'.format(split))
    
    #     dataloader, sampler = build_dataloader(cfg, split, is_train=False, val_distributed=False)
    #     per_ins_stat, per_cls_stat, _, _, others = evaluation(model=model.module, val_dataloader=dataloader, cfg=cfg, 
    #                                                 epoch=epoch, global_step=global_step, 
    #                                                 generate_cfg=cfg['testing']['cfg'],
    #                                                 save_dir=os.path.join(model_dir,split), return_prob=True)
    #     sync_results(per_ins_stat, per_cls_stat, save_dir=os.path.join(model_dir,split), wandb_run=None, sync=False)
    #     # if cfg['data']['dataset_name'] == 'WLASL_2000':
    #     #     for o in [1000, 300, 100]:
    #     #         logger.info('-----------------------Variant: {:d}-------------------------'.format(o))
    #     #         sync_results(others[str(o)]['per_ins_stat'], others[str(o)]['per_cls_stat'],
    #     #                     save_dir=os.path.join(model_dir,split), wandb_run=None, sync=False)
    # synchronize()

    # if is_main_process():
    #     split = 'test'
    #     logger.info('Evaluate on {} set'.format(split))
    #     dataloader, sampler = build_dataloader(cfg, split, is_train=False, val_distributed=False)
    #     per_ins_stat, per_cls_stat, _, _, others = evaluation(model=model.module, val_dataloader=dataloader, cfg=cfg, 
    #                                                 epoch=epoch, global_step=global_step, 
    #                                                 generate_cfg=cfg['testing']['cfg'],
    #                                                 save_dir=os.path.join(model_dir,split), return_prob=True, return_others=False)
    #     sync_results(per_ins_stat, per_cls_stat, save_dir=os.path.join(model_dir,split), wandb_run=None, sync=False)
    #     # if cfg['data']['dataset_name'] == 'WLASL_2000':
    #     #     other_vocab = [1000, 300, 100]
    #     # elif cfg['data']['dataset_name'] == 'MSASL_1000':
    #     #     other_vocab = [500, 200, 100]
    #     # else:
    #     #     other_vocab = []
    #     # for o in other_vocab:
    #     #     logger.info('-----------------------Variant: {:d}-------------------------'.format(o))
    #     #     sync_results(others[str(o)]['per_ins_stat'], others[str(o)]['per_cls_stat'],
    #     #                 save_dir=os.path.join(model_dir,split), wandb_run=None, sync=False)
    # synchronize()
    
    if wandb_run != None:
        wandb_run.finish()
    