import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from sys import platform
from logging import Logger
from typing import Callable, Optional
import numpy as np
import torch
from torch import nn, Tensor
import yaml
from torch.utils.tensorboard import SummaryWriter
import wandb
from collections import defaultdict
import math, pickle
DATASETS = ['phoenix','phoenix2014tsi','phoenix2014','phoenix2014si','csl','cslr','wlasl2000','tvb','phoenixcomb']

def schedule_value(y0, t, schedule):
    if schedule=='constant':
        return y0
    elif schedule=='linear':
        yt = y0*(1-t)
        return yt
    elif schedule=='cosine':
        yt = math.cos(t*(math.pi/2))
        return yt
    else:
        raise ValueError

def make_wandb(model_dir, cfg):
    if 'debug' not in model_dir and is_main_process():
        try:
            wandb.login(key='9451b6c734f487665f86afbd6143dc8db0ffda3f')
            run = wandb.init(project='TwoStream', 
                        #entity=model_dir, 
                        config=cfg, reinit=True)
            #wandb.run.name = model_dir.replace('experiments/','')
            wandb.run.name = '/'.join(model_dir.split('/')[-2:])
            wandb.run.save()
            return run
        except:
            return None
    else:
        return None

def neq_load_customized(model, pretrained_dict, verbose=False):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        print(list(model_dict.keys()))
        print('\n=======Check Weights Loading======')
        print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape==v.shape:
            tmp[k] = v
        else:
            if verbose:
                print(k)
    if verbose:
        print('---------------------------')
        print('Weights not loaded into new model:')
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
            elif model_dict[k].shape != pretrained_dict[k].shape:
                print(k, 'shape mis-matched, not loaded')
        print('===================================\n')


    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def upd_MAE_ckpt_keys(ckpt):
    #filter decoder
    ckpt_state_upd = dict((k.replace('core.encoder', 'recognition_network.visual_backbone_keypoint.backbone'), v) for (k, v) in ckpt.items() if 'encoder' in k)
    return ckpt_state_upd


def move_to_device(batch, device):
    for k, v in batch.items():
        if type(v)==dict:
            batch[k] = move_to_device(v, device)
        elif type(v)==torch.Tensor:
            batch[k] = v.to(device)
        elif type(v)==list and type(v[0])==torch.Tensor:
            batch[k] = [e.to(device) for e in v]
    return batch

def make_model_dir(model_dir: str, overwrite: bool = False) -> str:
    """
    Create a new directory for the model.
    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    # if is_main_process():
    #     if os.path.isdir(model_dir):
    #         if not overwrite:
    #             raise FileExistsError("Model directory exists and overwriting is disabled.")
    #         # delete previous directory to start with empty dir again
    #         shutil.rmtree(model_dir)
    #     os.makedirs(model_dir)
    if is_main_process():
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        elif overwrite:
            shutil.rmtree(model_dir)
            os.makedirs(model_dir)
    synchronize()
    return model_dir

def get_logger():
    return logger
    
def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.
    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    global logger
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(level=logging.DEBUG)
        fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)
        if platform == "linux":
            sh = logging.StreamHandler()
            if not is_main_process():
                sh.setLevel(logging.ERROR)
            sh.setFormatter(formatter)
            logging.getLogger("").addHandler(sh)
        return logger

def make_writer(model_dir):
    if is_main_process():
        writer = SummaryWriter(log_dir=os.path.join(model_dir + "/tensorboard/"))
    else:
        writer = None
    return writer

def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    Write configuration to log.
    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))

def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.
    :param seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.
    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    #check
    # if 'RecognitionNetwork' in cfg['model'] and 'resnet' in cfg['model']['RecognitionNetwork']:
    #     assert cfg['data'].get('transform_cfg','s3d')=='resnet', cfg['data'].get('transform_cfg','s3d')
    #deprecate

    #extend single source to multi-source (extend old code)
    if cfg['data'].get('multi', False)==False:
        datasetname = cfg['data']['dataset_name']
        new_datacfg = {
            'multi':True, 
            datasetname:cfg['data'],
            'input_data':cfg['data']['input_data'],
            'level': cfg['data']['level'],
            'dataset_name': cfg['data']['dataset_name']}
        cfg['data'] = new_datacfg
        cfg['datanames'] = [datasetname]
    else:
        input_data = [cfg['data'][k]['input_data'] for k in DATASETS if k in cfg['data']]
        assert len(set(input_data))==1, input_data
        cfg['data']['input_data'] = input_data[0]
        cfg['datanames'] = [k for k in DATASETS if k in cfg['data']]
    if 'RecognitionNetwork' in cfg['model']:
        # assert cfg['training']['batch_size']==1, cfg['training']['batch_size']
        input_streams, keypoints_nums = [], []
        from dataset.Dataset import get_keypoints_num
        for datasetname, data_cfg in cfg['data'].items():
            if not datasetname in DATASETS:
                continue
            input_stream = []
            if 'rgb' in data_cfg.get('input_streams', ['rgb']):
                input_stream.append('rgb')
            if 'keypoint' in data_cfg.get('input_streams', ['rgb']):
                input_stream.append('keypoint')
                assert 'keypoint_s3d' in cfg['model']['RecognitionNetwork'] or 'keypoint_resnet3d' in cfg['model']['RecognitionNetwork']
                keypoints_num = get_keypoints_num(
                    keypoint_file=data_cfg['keypoint_file'], use_keypoints=data_cfg['use_keypoints'])
                keypoints_nums.append(keypoints_num)
            else:
                keypoints_nums.append(0)
            input_streams.append(input_stream)
            if len(input_streams)>1:
                assert ''.join(input_stream)==''.join(input_streams[-1])
        # print(keypoints_nums)
        assert len(set(keypoints_nums))==1, keypoints_nums #unified
        cfg['data']['input_streams'] = input_streams[0]
        if 'keypoint' in cfg['data']['input_streams'] and  'keypoint_s3d' in cfg['model']['RecognitionNetwork']:
            cfg['model']['RecognitionNetwork']['keypoint_s3d']['in_channel'] = keypoints_num
            print(f'Overwrite cfg.model.RecognitionNetwork.keypoint_s3d.in_channel -> {keypoints_num}')
            #heatmap_cfg
            if 'raw_size' in cfg['model']['RecognitionNetwork']['heatmap_cfg']:
                heatmap_cfg = {**cfg['model']['RecognitionNetwork']['heatmap_cfg']}
                cfg['model']['RecognitionNetwork']['heatmap_cfg'] = defaultdict(lambda :heatmap_cfg)

        if 'keypoint_resnet3d' in cfg['model']['RecognitionNetwork']:
            cfg['model']['RecognitionNetwork']['keypoint_resnet3d']['in_channels'] = keypoints_num
            print(f'Overwrite cfg.model.RecognitionNetwork.keypoint_resnet3d.in_channels -> {keypoints_num}')
    return cfg

def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None
    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint

def load_checkpoint(path: str, map_location: str='cpu') -> dict:
    """
    Load model from saved checkpoint.
    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint

def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training
    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    os.system('cp {} {}'.format(target, link_name))
    # try:
    #     os.symlink(target, link_name)
    # except FileExistsError as e:
    #     if e.errno == errno.EEXIST:
    #         os.remove(link_name)
    #         os.symlink(target, link_name)
    #     else:
    #         raise e

def is_main_process():
    return 'WORLD_SIZE' not in os.environ or os.environ['WORLD_SIZE']=='1' or os.environ['LOCAL_RANK']=='0'

def init_DDP():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:{}'.format(local_rank))
    torch.distributed.init_process_group(backend="nccl", init_method="env://")    
    return local_rank, int(os.environ['WORLD_SIZE']), device

def synchronize():
    torch.distributed.barrier()

def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training
    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False

def merge_pkls(path, split, from_ckpt=False):
    final = {}
    # if split == 'dev':
    #     # check val first
    #     for fname in os.listdir(path):
    #         if 'val' in fname and fname != 'val.pkl':
    #             with open(os.path.join(path, fname), 'rb') as f:
    #                 data = pickle.load(f)
    #             final.update(data)
    for fname in os.listdir(path):
        if split in fname and fname != '{:s}.pkl'.format(split):
            with open(os.path.join(path, fname), 'rb') as f:
                data = pickle.load(f)
            final.update(data)

    if not from_ckpt:
        with open(path+'/{:s}.pkl'.format(split), 'wb') as f:
            pickle.dump(final, f)
        print("Merged to {:s}/{:s}.pkl".format(path, split))
    else:
        with open(path+'/{:s}_ckpt.pkl'.format(split), 'wb') as f:
            pickle.dump(final, f)
        print("Merged to {:s}/{:s}_ckpt.pkl".format(path, split))