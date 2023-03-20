import warnings, argparse, os, sys, shutil, time, queue
sys.path.append(os.getcwd())#slt dir
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from modelling.model import build_model
from utils.optimizer import build_optimizer, build_scheduler
from utils.progressbar import ProgressBar
warnings.filterwarnings("ignore")
from utils.misc import (
    load_config,
    make_model_dir,
    make_logger, make_writer, make_wandb,
    set_seed,
    is_main_process, init_DDP,
    synchronize 
)
from dataset.Dataloader import build_dataloader
from prediction import evaluation
import wandb

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
    eval_results = evaluation(
        model=model, val_dataloader=val_dataloader, cfg=cfg, 
        tb_writer=tb_writer, wandb_run=wandb_run,
        epoch=epoch, global_step=global_step, generate_cfg=generate_cfg,
        save_dir=os.path.join(cfg['training']['model_dir'],'validation',tag),
        do_recognition=cfg['task'] not in ['G2T','S2T_glsfree'],
        do_translation=cfg['task']!='S2G')
    metric = 'bleu4' if '2T' in cfg['task'] else 'wer'
    if metric=='bleu4':
        score = eval_results['bleu']['bleu4']
        best_score = max(best_score, score)
    elif metric=='wer':
        score = eval_results['wer']
        best_score = min(best_score, score)
    logger.info('best_score={:.2f}'.format(best_score))
    ckpt_file = save_model(model=model, optimizer=optimizer, scheduler=scheduler,
        output_file=os.path.join(cfg['training']['model_dir'],'ckpts',tag+'.ckpt'),
        epoch=epoch, global_step=global_step,
        current_score=score)

    if best_score==score:
        os.system('cp {} {}'.format(ckpt_file, os.path.join(cfg['training']['model_dir'],'ckpts','best.ckpt')))
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

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--wandb", 
        action="store_true", 
        help='turn on wandb'
    )
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    cfg['local_rank'], cfg['world_size'], cfg['device'] = init_DDP()
    set_seed(seed=cfg["training"].get("random_seed", 42))    
    model_dir = make_model_dir(
        model_dir=cfg['training']['model_dir'], 
        overwrite=cfg['training'].get('overwrite',False))
    global logger
    logger = make_logger(
        model_dir=model_dir,
        log_file='train.rank{}.log'.format(cfg['local_rank']))
    tb_writer = make_writer(model_dir=model_dir) 
    if args.wandb:
        wandb_run = make_wandb(model_dir=model_dir, cfg=cfg)
    else:
        wandb_run = None
    if is_main_process():
        os.system('cp {} {}/'.format(args.config, model_dir))
    synchronize()

    model = build_model(cfg)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('# Total parameters = {}'.format(total_params))
    logger.info('# Total trainable parameters = {}'.format(total_params_trainable))

    model = DDP(model, 
        device_ids=[cfg['local_rank']], 
        output_device=cfg['local_rank'],
        find_unused_parameters=True)

    train_dataloader, train_sampler = build_dataloader(cfg, 'train', model.module.text_tokenizer, model.module.gloss_tokenizer)
    dev_dataloader, dev_sampler = build_dataloader(cfg, 'dev', model.module.text_tokenizer, model.module.gloss_tokenizer)

    optimizer = build_optimizer(config=cfg['training']['optimization'], model=model.module) 
    scheduler, scheduler_type = build_scheduler(config=cfg['training']['optimization'], optimizer=optimizer)
    assert scheduler_type=='epoch'
    start_epoch, total_epoch, global_step = 0, cfg['training']['total_epoch'], 0
    val_unit, val_freq = cfg['training']['validation']['unit'], cfg['training']['validation']['freq']
    global ckpt_queue, best_score
    ckpt_queue = queue.Queue(maxsize=cfg['training']['keep_last_ckpts'])
    best_score = -100 if '2T' in cfg['task'] else 10000 

    #RESUME TRAINING
    if cfg['training'].get('from_ckpt', False):
        synchronize()
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

        torch.manual_seed(cfg["training"].get("random_seed", 42)+start_epoch)
        train_dataloader, train_sampler = build_dataloader(cfg, 'train', model.module.text_tokenizer, model.module.gloss_tokenizer)
        dev_dataloader, dev_sampler = build_dataloader(cfg, 'dev', model.module.text_tokenizer, model.module.gloss_tokenizer)
        logger.info('Sucessfully resume training from {:s}'.format(latest_ckpt))
        

    if is_main_process():
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        tb_writer = SummaryWriter(log_dir=os.path.join(model_dir,"tensorboard"))
    else:
        pbar, tb_writer = None, None
        
    for epoch_no in range(start_epoch, total_epoch):
        train_sampler.set_epoch(epoch_no)
        logger.info('Epoch {}, Training examples {}'.format(epoch_no, len(train_dataloader.dataset)))
        scheduler.step()
        for step, batch in enumerate(train_dataloader):
            model.module.set_train()
            output = model(is_train=True, **batch)

            with torch.autograd.set_detect_anomaly(True):           
                output['total_loss'].backward()
            optimizer.step()
            model.zero_grad()

            if is_main_process() and tb_writer:
                for k,v in output.items():
                    if '_loss' in k:
                        tb_writer.add_scalar('train/'+k, v, global_step)
                lr = scheduler.optimizer.param_groups[0]["lr"]
                tb_writer.add_scalar('train/learning_rate', lr, global_step)
                if wandb_run!=None:
                    wandb.log({k: v for k,v in output.items() if '_loss' in k})
                    wandb.log({'learning_rate': lr})
            if is_main_process() and val_unit=='step' and global_step%val_freq==0 and global_step>0:
                evaluate_and_save(
                    model=model.module, optimizer=optimizer, scheduler=scheduler,
                    val_dataloader=dev_dataloader,
                    cfg=cfg, tb_writer=tb_writer, wandb_run=wandb_run,
                    global_step=global_step,
                    generate_cfg=cfg['training']['validation']['cfg'])
            global_step += 1
            if pbar:
                pbar(step)
                
        if is_main_process() and val_unit=='epoch' and epoch_no%val_freq==0 and epoch_no>0:
            evaluate_and_save(
                model=model.module, optimizer=optimizer, scheduler=scheduler,
                val_dataloader=dev_dataloader,
                cfg=cfg, tb_writer=tb_writer, wandb_run=wandb_run,
                epoch=epoch_no,
                generate_cfg=cfg['training']['validation']['cfg'])
        print()    
          

    #test
    if is_main_process():
        load_model_path = os.path.join(cfg['training']['model_dir'],'ckpts','best.ckpt')
        state_dict = torch.load(load_model_path, map_location='cuda')
        model.module.load_state_dict(state_dict['model_state'])
        epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
        logger.info('Load model ckpt from '+load_model_path)
        do_translation, do_recognition = cfg['task']!='S2G', cfg['task']!='G2T' 
        for split in ['dev','test']:
            logger.info('Evaluate on {} set'.format(split))
            dataloader, sampler = build_dataloader(cfg, split, model.module.text_tokenizer, model.module.gloss_tokenizer)
            evaluation(model=model.module, val_dataloader=dataloader, cfg=cfg, 
                    epoch=epoch, global_step=global_step, 
                    generate_cfg=cfg['testing']['cfg'],
                    save_dir=os.path.join(model_dir,split),
                    do_translation=do_translation, do_recognition=do_recognition)   
    if wandb_run!=None:
        wandb_run.finish()    

        


    

