import errno
import functools
import getpass
import os
import pickle as pkl
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io
import torch


def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


@functools.lru_cache(maxsize=64, typed=False)
def load_checkpoint(ckpt_path):
    return torch.load(ckpt_path, map_location={"cuda:0": "cpu"})


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError(f"Cannot convert {type(ndarray)} to torch tensor")
    return ndarray


def save_checkpoint(
    state, checkpoint="checkpoint", filename="checkpoint.pth.tar", snapshot=1
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state["epoch"] % snapshot == 0:
        dest = os.path.join(checkpoint, "checkpoint_%03d.pth.tar" % state["epoch"])
        shutil.copyfile(filepath, dest)


def load_checkpoint_flexible(model, optimizer, args, plog):
    msg = f"no pretrained model found at {args.pretrained}"
    assert Path(args.pretrained).exists(), msg
    plog.info(f"=> loading checkpoint '{args.pretrained}'")
    checkpoint = load_checkpoint(args.pretrained)

    # This part handles ignoring the last layer weights if there is mismatch
    partial_load = False
    if "state_dict" in checkpoint:
        pretrained_dict = checkpoint["state_dict"]
    else:
        plog.info("State_dict key not found, attempting to use the checkpoint:")
        pretrained_dict = checkpoint

    # If the pretrained model is not torch.nn.DataParallel(model), append 'module.' to keys.
    if "module" not in sorted(pretrained_dict.keys())[0]:
        plog.info('Appending "module." to pretrained keys.')
        pretrained_dict = dict(("module." + k, v) for (k, v) in pretrained_dict.items())

    model_dict = model.state_dict()

    for k, v in pretrained_dict.items():
        if not ((k in model_dict) and v.shape == model_dict[k].shape):
            plog.info(f"Unused from pretrain {k}")
            partial_load = True

    for k, v in model_dict.items():
        if k not in pretrained_dict:
            plog.info(f"Missing in pretrain {k}")
            partial_load = True

    if args.init_cross_language != "":
        # HACK: get the classifier weights before throwing them away
        from utils.cross_language import get_classification_params

        pretrained_w, pretrained_b = get_classification_params(
            pretrained_dict, arch=args.arch
        )

    if partial_load:
        plog.info("Removing or not initializing some layers...")
        # 1. filter out unnecessary keys
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if (k in model_dict) and (v.shape == model_dict[k].shape)
        }

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # CAUTION: Optimizer not initialized with the pretrained one

        # Modification TO MAKE THE BSL ASL correspond
        if args.init_cross_language != "":
            from utils.cross_language import init_cross_language

            # NOTE: Update from Samuel that preserves previous behaviour if asl_dataset
            # is not set
            asl_dataset = getattr(args, "asl_dataset", args.datasetname)

            model = init_cross_language(
                init_str=args.init_cross_language,
                model=model,
                pretrained_w=pretrained_w,
                pretrained_b=pretrained_b,
                asl_dataset=asl_dataset,
                bsl_pkl=args.word_data_pkl,
            )
    else:
        plog.info("Loading state dict.")
        model.load_state_dict(checkpoint["state_dict"])
        plog.info("Loading optimizer.")
        optimizer.load_state_dict(checkpoint["optimizer"])

    del checkpoint, pretrained_dict
    # if args.featurize_mode:
    #     assert not partial_load, "Must use full weights for featurization!"
    return partial_load


def save_pred(preds, checkpoint="checkpoint", filename="preds_valid.mat", meta=None):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    mdict = {"preds": preds}
    if meta is not None:
        mdict.update(meta)
    print(f"Saving to {filepath}")
    scipy.io.savemat(filepath, mdict=mdict, do_compression=False, format="4")


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma, num_gpus=1, warmup=5):
    """Sets the learning rate to the initial LR decayed by schedule.

    Use linear warmup for multi-gpu training: https://arxiv.org/abs/1706.02677
    """
    if epoch in schedule:
        lr *= gamma
        if num_gpus > 1 and epoch < warmup:
            param_lr = lr * (epoch + 1) / warmup
        else:
            param_lr = lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_lr
    return lr


# Show num_figs equi-distant images
# If the epoch is too small, show all
def is_show(num_figs, iter_no, epoch_len):
    if num_figs == 0:
        return 0
    show_freq = int(epoch_len / num_figs)
    if show_freq != 0:
        return iter_no % show_freq == 0
    else:
        return 1


class Timer:
    """A simple timing utility."""

    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
