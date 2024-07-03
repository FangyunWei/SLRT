from logging import Logger
from typing import DefaultDict, Text
import warnings
import pickle, gzip
from google.protobuf.reflection import ParseMessage
from collections import defaultdict
from modelling.model import build_model
from modelling.recognition import ctc_decode_func
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
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from utils.misc import (
    get_logger,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger, make_writer,
    set_seed,
    symlink_update,
    is_main_process, init_DDP, move_to_device,
    synchronize 
)
from dataset.Dataloader import build_dataloader
from utils.progressbar import ProgressBar
from utils.metrics import bleu, rouge, wer_list
from utils.phoenix_cleanup import clean_phoenix_2014_trans

def extract_visual_feature(model, dataloader, cfg):  #to-do output_dir
    logger = get_logger()
    if 'output_feature' not in cfg['testing']:
        if cfg['task'] == 'S2G':
            suffix2data = {'head_rgb_input':[],'head_keypoint_input':[]}
        elif cfg['task'] in ['S2T','G2T']:
            suffix2data = {'inputs_embeds':[]}
        else:
            raise ValueError
    else:
        suffix2data = {}
        for k in cfg['testing']['output_feature']:
            suffix2data[k] = []
    logger.info('Extract features: '+' '.join(list(suffix2data.keys())))
    pbar = ProgressBar(n_total=len(dataloader), desc='Extract_feature')
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            #forward -- loss
            batch = move_to_device(batch, cfg['device'])
            forward_output = model(is_train=False, **batch)
            entry = {}
            if cfg['task'] == 'S2G':
                assert cfg['training']['batch_size']==1, cfg['training']['batch_size']
            for ii in range(len(batch['name'])):
                for key in ['name','gloss','text','num_frames']:
                    entry[key] = batch[key][ii]
                for suffix, output_data in suffix2data.items():
                    if suffix in forward_output:
                        output_data.append({**entry, 
                            'sign':forward_output[suffix][ii].detach().cpu()})
                    if suffix == 'inputs_embeds':
                        #consider mask
                        valid_len = torch.sum(forward_output['transformer_inputs']['attention_mask'][ii])
                        output_data.append({**entry, 
                            'sign':forward_output['transformer_inputs'][suffix][ii,:valid_len].detach().cpu()})
            pbar(step)
    return suffix2data


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    # parser.add_argument(
    #     "--outputdir",
    #     default='prediction',
    #     type=str
    # )
    parser.add_argument(
        '--ckpt_path',
        default='',
        type=str
    )
    # parser.add_argument(
    #     '--output_feature',
    #     default='s3d_pooled',
    #     type=str,
    #     help='sep by ,'
    # )
    parser.add_argument(
        '--output_split',
        default='dev,test,train',
        type=str,
        help='sep by ,'
    )
    parser.add_argument(
        '--output_subdir',
        default='extract_feature',
        type=str
    )
    parser.add_argument(
        '--dataset',
        default='phoenix',
        type=str
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg['local_rank'], cfg['world_size'], cfg['device'] = init_DDP()
    args.outputdir=os.path.join(cfg['training']['model_dir'],args.output_subdir)
    #cfg['testing']['output_feature'] = args.output_feature.split(',')
    if int(cfg['local_rank'])==0:
        os.makedirs(args.outputdir, exist_ok=True)
        os.system('cp {} {}/'.format(args.config, args.outputdir))
    synchronize()
    logger = make_logger(model_dir=args.outputdir, log_file=f"prediction.log.{cfg['local_rank']}")
    model = build_model(cfg)
    #load model
    if args.ckpt_path=='':
        args.ckpt_path = os.path.join(cfg['training']['model_dir'], 'ckpts/{}_best.ckpt'.format(cfg['data']['dataset_name']))
    if not args.ckpt_path=='':
        if os.path.isfile(args.ckpt_path):
            load_model_path = os.path.join(args.ckpt_path)
            state_dict = torch.load(load_model_path, map_location='cuda')
            model.load_state_dict(state_dict['model_state'])
            logger.info('Load model ckpt from '+load_model_path)
        else:
            logger.info(f'{args.ckpt_path} does not exist, model from scratch')

    model = DDP(model, 
        device_ids=[cfg['local_rank']], 
        output_device=cfg['local_rank'])
    for split in args.output_split.split(','):
        logger.info('Extract visual feature on {} set'.format(split))
        dataloader, sampler = build_dataloader(cfg, split, model.module.text_tokenizer, model.module.gloss_tokenizer,
                mode='eval', val_distributed=True)            
        results = extract_visual_feature(model.module, dataloader, cfg)
        for name, output_data in results.items():
            subdir = os.path.join(args.outputdir, name)
            os.makedirs(subdir, exist_ok=True)
            outputfile = os.path.join(subdir, f"{split}.pkl.rank{cfg['local_rank']}")
            with gzip.open(outputfile, 'wb') as f:
                pickle.dump(output_data, f)
            logger.info('Save feature as '+outputfile)

        synchronize()
        if int(cfg['local_rank'])==0:
            #gather
            for key in results:
                subdir = os.path.join(args.outputdir, key)
                name2data = {}
                for rank in range(cfg['world_size']):
                    split_file = os.path.join(subdir, f"{split}.pkl.rank{rank}")
                    with gzip.open(split_file, 'rb') as f:
                        split_data = pickle.load(f)
                    for d in split_data:
                        name2data[d['name']] = d
                gather_file = os.path.join(subdir, f"{split}.pkl")
                with gzip.open(gather_file,'wb') as f:
                    pickle.dump([v for k, v in name2data.items()], f)
                logger.info(f'Gather {split} -> {len(name2data)}')
                logger.info(f'save as {gather_file}')            
