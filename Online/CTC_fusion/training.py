import warnings
from modelling.model import build_model
from utils.optimizer import build_optimizer, build_scheduler
warnings.filterwarnings("ignore")
import argparse
import numpy as np
from glob import glob
from copy import deepcopy
import os, sys, pickle
import shutil
import time
import queue
sys.path.append(os.getcwd())#slt dir
import torch
from torch.nn.parallel import DistributedDataParallel as DDP, distributed
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from utils.misc import (
    load_config,
    log_cfg,
    load_checkpoint, neq_load_customized,
    make_model_dir,
    make_logger, make_writer, make_wandb,
    schedule_value,
    set_seed,
    symlink_update,
    is_main_process, init_DDP,
    synchronize 
)
from dataset.Dataloader import build_dataloader
from utils.progressbar import ProgressBar
from prediction import evaluation
import wandb
from collections import defaultdict
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
        'current_score': dict(current_score) if current_score!=None else 0,
    }
    torch.save(state, output_file)
    logger.info('Save model state as '+ output_file)
    return output_file

def evaluate_and_save(model, optimizer, scheduler, val_dataloader, cfg, 
        tb_writer, wandb_run=None,
        epoch=None, global_step=None, generate_cfg={}, only_save=False):
    tag = 'epoch_{:02d}'.format(epoch) if epoch!=None else 'step_{}'.format(global_step)
    #save
    global best_score, ckpt_queue
    if not only_save:
        datasetname2eval_results = evaluation(
            model=model, val_dataloader=val_dataloader, cfg=cfg, 
            tb_writer=tb_writer, wandb_run=wandb_run,
            epoch=epoch, global_step=global_step, generate_cfg=generate_cfg,
            save_dir=os.path.join(cfg['training']['model_dir'],'validation',tag),
            do_recognition=cfg['task'] not in ['G2T','S2T_glsfree'],
            do_translation=cfg['task']!='S2G')
    else:
        datasetname2eval_results = None
    ckpt_file = save_model(model=model, optimizer=optimizer, scheduler=scheduler,
        output_file=os.path.join(cfg['training']['model_dir'],'ckpts',tag+'.ckpt'),
        epoch=epoch, global_step=global_step,
        current_score=datasetname2eval_results)
    
    if not only_save:
        for datasetname, eval_results in datasetname2eval_results.items():
            metric = 'bleu4' if '2T' in cfg['task'] else 'wer'
            if metric=='bleu4':
                score = eval_results['bleu']['bleu4']
                best_score = max(best_score, score)
            elif metric=='wer':
                score = eval_results['wer']
                best_score = min(best_score, score)
            logger.info('{} new_score={:.2f} best_score={:.2f}'.format(datasetname, score, best_score))

            if best_score==score:
                os.system('cp {} {}'.format(ckpt_file, os.path.join(cfg['training']['model_dir'],'ckpts',f'{datasetname}_best.ckpt')))
            if ckpt_queue[datasetname].full():
                to_delete = ckpt_queue[datasetname].get()
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    logger.warning(
                        "Wanted to delete old checkpoint %s but " "file does not exist.",
                        to_delete,
                    )
            ckpt_queue[datasetname].put(ckpt_file)        

    
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
    set_seed(seed=cfg["training"].get("random_seed", 42))    
    model_dir = make_model_dir(
        model_dir=cfg['training']['model_dir'], 
        overwrite=cfg['training'].get('overwrite',False))
    global logger
    logger = make_logger(
        model_dir=model_dir,
        log_file='train.rank{}.log'.format(cfg['local_rank']))
    tb_writer = make_writer(model_dir=model_dir) #SummaryWriter or None (local_rank>=1)
    wandb_run = make_wandb(model_dir=model_dir, cfg=cfg)
    if is_main_process():
        os.system('cp {} {}/'.format(args.config, model_dir))
        os.system('cp -r modelling/* {}/'.format(model_dir))
        os.system('cp -r dataset/* {}/'.format(model_dir))
    synchronize()
    #MODEL
    #!!! Must-do sync_batch
    model = build_model(cfg)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('# Total parameters = {}'.format(total_params))
    logger.info('# Total trainable parameters = {}'.format(total_params_trainable))
    #logger.info('Trainable parameters: '+', '.join([n for n, p in model.named_parameters() if p.requires_grad]))
    
    # if 'load_ckpt' in cfg['training']:
    #     neq_load_customized(model, 
    #         pretrained_dict=torch.load(cfg['training']['load_ckpt'], map_location='cpu')['model_state'])
    #     logger.info('Load ckpt from '+cfg['training']['load_ckpt'])
    #load pretrained ckpt
    s2t_ckpt_file = cfg['training'].get('load_s2t_ckpt', None)
    if s2t_ckpt_file and cfg['task'] in ['S2T']:
        pretrained_dict = torch.load(s2t_ckpt_file, map_location='cpu')['model_state']
        neq_load_customized(model, pretrained_dict, verbose=False)
        logger.info("Load S2T ckpt {:s}".format(s2t_ckpt_file))

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
        else:
            updated_dict = pretrained_dict.copy()
        del pretrained_dict
        neq_load_customized(model, updated_dict, verbose=False)
        logger.info("Load ckpt {:s}".format(ckpt_file))

    model = DDP(model, 
        device_ids=[cfg['local_rank']], 
        output_device=cfg['local_rank'],
        find_unused_parameters=True)
    #DATASET
    train_dataloader, train_sampler = build_dataloader(cfg, 'train', model.module.text_tokenizer, model.module.gloss_tokenizer)
    dev_dataloader, dev_sampler = build_dataloader(cfg, 'dev', model.module.text_tokenizer, model.module.gloss_tokenizer)

    #OPTIMIZATION
    optimizer = build_optimizer(config=cfg['training']['optimization'], model=model.module) #assign different learning rates for different modules
    scheduler, scheduler_type = build_scheduler(config=cfg['training']['optimization'], optimizer=optimizer)
    assert scheduler_type=='epoch'
    start_epoch, total_epoch, global_step = 0, cfg['training']['total_epoch'], 0
    pseudo_epoch = 0
    val_unit, val_freq = cfg['training']['validation']['unit'], cfg['training']['validation']['freq']
    global ckpt_queue, best_score
    ckpt_queue = defaultdict(lambda: queue.Queue(maxsize=cfg['training']['keep_last_ckpts']))
    best_score = -100 if '2T' in cfg['task'] else 10000
    # best_score = defaultdict(lambda: -100) if '2T' in cfg['task'] else defaultdict(lambda: 10000)

    #RESUME TRAINING
    if cfg['training'].get('from_ckpt', False):
        synchronize()
        if os.path.exists(os.path.join(model_dir, 'ckpts')):
            ckpt_lst = sorted([f for f in os.listdir(os.path.join(model_dir, 'ckpts')) if 'best' not in f])
            if len(ckpt_lst) > 0:
                latest_ckpt = ckpt_lst[-1]
                latest_ckpt = os.path.join(model_dir, 'ckpts', latest_ckpt)
                state_dict = torch.load(latest_ckpt, 'cuda:{:d}'.format(cfg['local_rank']))
                model.module.load_state_dict(state_dict['model_state'])
                optimizer.load_state_dict(state_dict['optimizer_state'])
                scheduler.load_state_dict(state_dict['scheduler_state'])
                start_epoch = state_dict['epoch']+1 if state_dict['epoch'] is not None else int(latest_ckpt.split('_')[-1][:-5])+1
                global_step = state_dict['global_step']+1 if state_dict['global_step'] is not None else 0
                best_score = state_dict['best_score'] #a dict of a float
                # if type(best_score)!=dict: #extend old code to new code
                #     if len(cfg['datanames'])==1:
                #         best_score = {cfg['datanames'][0]: best_score}
                #     else: #old code to new code the best_score is an average (reset)
                #         best_score = defaultdict(lambda: -100) if '2T' in cfg['task'] else defaultdict(lambda: 10000)

                # change dataloader order
                torch.manual_seed(cfg["training"].get("random_seed", 42)+start_epoch)
                train_dataloader, train_sampler = build_dataloader(cfg, 'train', model.module.text_tokenizer, model.module.gloss_tokenizer)
                dev_dataloader, dev_sampler = build_dataloader(cfg, 'dev', model.module.text_tokenizer, model.module.gloss_tokenizer)
                logger.info('Sucessfully resume training from {:s}'.format(latest_ckpt))
                print(best_score)
        
    #to-do start_epoch
    if 'RecognitionNetwork' in cfg['model'] and 'online_augmentation' in cfg['model']['RecognitionNetwork']:
        online_augmentation = True
        memory_bank_local_dir = os.path.join(cfg['training']['model_dir'],'memory_bank','local')
        memory_bank_global_dir = os.path.join(cfg['training']['model_dir'],'memory_bank','global')
        augmentation_start_epoch = cfg['model']['RecognitionNetwork']['online_augmentation']['start_epoch']
        pseudo_ratio = cfg['model']['RecognitionNetwork']['online_augmentation']['pseudo_ratio']
        pseudo_schedule = cfg['model']['RecognitionNetwork']['online_augmentation'].get('pseudo_schedule', 'constant')
        load_memory_bank_file = cfg['model']['RecognitionNetwork']['online_augmentation'].get('load_bank',None)
        if is_main_process():
            os.makedirs(memory_bank_global_dir, exist_ok=True)
            os.makedirs(memory_bank_local_dir, exist_ok=True)
        synchronize()
    else:
        online_augmentation = False

    #Others (pbar, tb)
    if is_main_process():
        if os.environ.get('enable_pbar', '1')=='1':
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        else:
            pbar = None
        # tb_writer = SummaryWriter(log_dir=os.path.join(model_dir,"tensorboard"))
        tb_writer = None
    else:
        pbar, tb_writer = None, None
        
    # logger.info('Evaluation at the beginning ...')
    # if is_main_process():
    #     evaluate_and_save(
    #         model=model.module, optimizer=optimizer, scheduler=scheduler,
    #         val_dataloader=dev_dataloader,
    #         cfg=cfg, tb_writer=None, wandb_run=wandb_run, epoch=start_epoch-1,
    #         generate_cfg=cfg['training']['validation']['cfg'])  

    #iteration
    for epoch_no in range(start_epoch, total_epoch):
        train_sampler.set_epoch(epoch_no)
        logger.info('Epoch {}, Training examples {}'.format(epoch_no, len(train_dataloader.dataset)))
        scheduler.step()
        if online_augmentation:
            memory_bank_local = {dn:defaultdict(list) for dn in cfg['datanames']}
            if epoch_no>=augmentation_start_epoch: #reset dataloader
                if epoch_no==augmentation_start_epoch and load_memory_bank_file!=None:
                    assert len(glob(load_memory_bank_file))
                else:
                    load_memory_bank_file = os.path.join(memory_bank_global_dir, f'*_epoch{epoch_no-1}.pkl')
                logger.info(f'Load memory bank at epoch {epoch_no} from {glob(load_memory_bank_file)}...')
                datasetname2memory_bank = {}
                for datasetname in cfg['datanames']:
                    with open(load_memory_bank_file.replace('*',datasetname),'rb') as f:
                        datasetname2memory_bank[datasetname] = pickle.load(f)
                pseudo_epoch = (epoch_no-augmentation_start_epoch)/(total_epoch-1-augmentation_start_epoch)
                cur_pseudo_ratio = schedule_value(
                    y0=pseudo_ratio, t=pseudo_epoch, 
                    schedule=pseudo_schedule)
                train_dataloader, train_sampler = train_dataloader.dataset.set_pseudo(
                    ratio=cur_pseudo_ratio, dataloader=train_dataloader, memory_bank=datasetname2memory_bank
                )
                

            
        for step, batch in enumerate(train_dataloader):
            if 'debug' in model_dir and step == 1:
                break
            # print(batch['gloss'])
            model.module.set_train()
            # if 'recognition_inputs' in batch:  #'recognition_inputs']['selected_indexs'
            #     selected_indexes = batch['recognition_inputs'].pop('selected_indexs',None)
            #     print(selected_indexes)
            if 'recognition_inputs' in batch:
                batch['recognition_inputs']['pseudo_epoch'] = pseudo_epoch
            output = model(is_train=True, **batch)
            if online_augmentation and batch['name'][0]!='pseudo': #gather memory_bank_local
                for gls, ls in output['pseudo_boundaries'].items():
                    memory_bank_local[batch['datasetname']][gls].extend(ls)
            #optimize
            #print(os.environ['LOCAL_RANK'],'forward')
            with torch.autograd.set_detect_anomaly(True):           
                output['total_loss'].backward()
            # print(os.environ['LOCAL_RANK'],'backward')
            optimizer.step()
            model.zero_grad()
            #visualization and logging
            if is_main_process() and tb_writer:
                for k,v in output.items():
                    if '_loss' in k:
                        tb_writer.add_scalar('train/'+k, v, global_step)
                lr = scheduler.optimizer.param_groups[0]["lr"]
                tb_writer.add_scalar('train/learning_rate', lr, global_step)
                if wandb_run!=None:
                    wandb.log({k: v for k,v in output.items() if '_loss' in k})
                    wandb.log({'learning_rate': lr})
                    if 'pseudo_weight' in output:
                        wandb.log({'pseudo_weight': output['pseudo_weight']})
            #validation and save (last ckpt and the best ckpt)
            if is_main_process() and val_unit=='step' and global_step%val_freq==0:
                evaluate_and_save(
                    model=model.module, optimizer=optimizer, scheduler=scheduler,
                    val_dataloader=dev_dataloader,
                    cfg=cfg, tb_writer=tb_writer, wandb_run=wandb_run,
                    global_step=global_step,
                    generate_cfg=cfg['training']['validation']['cfg'],
                    only_save=False)
            #DEBUG!



            #misc
            global_step += 1
            if pbar:
                pbar(step)
        

        if online_augmentation:#save memory_bank
            for datasetname, glsdic in memory_bank_local.items():
                outputfile = os.path.join(memory_bank_local_dir,f'{datasetname}_epoch{epoch_no}_rank{cfg["local_rank"]}.pkl')
                logger.info('Save as '+outputfile)
                with open(outputfile,'wb') as f:
                    pickle.dump(glsdic, f)
            synchronize()
            if is_main_process():
                logger.info('Main process, gather memory bank from all processes')
                for datasetname in cfg['datanames']:
                    glsdic_global = defaultdict(list)
                    n_samples = 0
                    for filename in glob(os.path.join(memory_bank_local_dir,f'{datasetname}_epoch{epoch_no}_rank*.pkl')):
                        with open(filename, 'rb') as f:
                            glsdic = pickle.load(f)
                        for gls, ls in glsdic.items():
                            glsdic_global[gls].extend(ls)
                            n_samples += len(ls)
                    outputfile = os.path.join(memory_bank_global_dir,f'{datasetname}_epoch{epoch_no}.pkl')
                    logger.info(f'Save as {outputfile}, #vocab={len(glsdic_global)}, #samples={n_samples}')
                    if wandb_run!=None:
                        wandb.log({f'{datasetname}_pseudo_vocab': len(glsdic_global)})
                        wandb.log({f'{datasetname}_pseudo_sample': n_samples})
                    with open(outputfile,'wb') as f:
                        pickle.dump(glsdic_global, f)
            synchronize()
        if is_main_process():
            evaluate_and_save(
                model=model.module, optimizer=optimizer, scheduler=scheduler,
                val_dataloader=dev_dataloader,
                cfg=cfg, tb_writer=tb_writer, wandb_run=wandb_run,
                epoch=epoch_no,
                generate_cfg=cfg['training']['validation']['cfg'],
                only_save=not (val_unit=='epoch' and epoch_no%val_freq==0))
        synchronize()
        print()    
    if is_main_process():
        evaluate_and_save(
            model=model.module, optimizer=optimizer, scheduler=scheduler,
            val_dataloader=dev_dataloader,
            cfg=cfg, tb_writer=None, wandb_run=wandb_run,
            epoch=epoch_no,
            generate_cfg=cfg['training']['validation']['cfg'])   



    #test
    if is_main_process():
        do_translation, do_recognition = cfg['task']!='S2G', cfg['task']!='G2T' #(and recognition loss>0 if S2T)
        for datasetname in cfg['datanames']:
            logger.info('Evaluate '+datasetname)
            load_model_path = os.path.join(cfg['training']['model_dir'],'ckpts',f'{datasetname}_best.ckpt')
            if not os.path.isfile(load_model_path):
                load_model_path = os.path.join(cfg['training']['model_dir'],'ckpts','best.ckpt')
            if os.path.isfile(load_model_path):
                state_dict = torch.load(load_model_path, map_location='cuda')
                model.module.load_state_dict(state_dict['model_state'])
                epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
                logger.info('Load model ckpt from '+load_model_path)
            else:
                logger.info(f'{load_model_path} does not exist')
                epoch, global_step = 0, 0
            cfg_ = deepcopy(cfg)
            cfg_['datanames'] = [datasetname]
            cfg_['data'] = {k:v for k,v in cfg['data'].items() if not k in cfg['datanames'] or k==datasetname}
            for split in ['dev','test']:
                logger.info('Evaluate on {} set'.format(split))
                dataloader, sampler = build_dataloader(cfg_, split, model.module.text_tokenizer, model.module.gloss_tokenizer)
                evaluation(model=model.module, val_dataloader=dataloader, cfg=cfg_, 
                        epoch=epoch, global_step=global_step, 
                        generate_cfg=cfg_['testing']['cfg'],
                        save_dir=os.path.join(model_dir,split),
                        do_translation=do_translation, do_recognition=do_recognition)
        '''
        #single dataset
        #load model
        load_model_path = os.path.join(cfg['training']['model_dir'],'ckpts','best.ckpt')
        state_dict = torch.load(load_model_path, map_location='cuda')
        model.module.load_state_dict(state_dict['model_state'])
        epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
        logger.info('Load model ckpt from '+load_model_path)
        do_translation, do_recognition = cfg['task']!='S2G', cfg['task']!='G2T' #(and recognition loss>0 if S2T)
        for split in ['dev','test']:
            logger.info('Evaluate on {} set'.format(split))
            dataloader, sampler = build_dataloader(cfg, split, model.module.text_tokenizer, model.module.gloss_tokenizer)
            evaluation(model=model.module, val_dataloader=dataloader, cfg=cfg, 
                    epoch=epoch, global_step=global_step, 
                    generate_cfg=cfg['testing']['cfg'],
                    save_dir=os.path.join(model_dir,split),
                    do_translation=do_translation, do_recognition=do_recognition)   
        '''
    if wandb_run!=None:
        wandb_run.finish()    

        


    

