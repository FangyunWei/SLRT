import os
import time

import numpy as np
import torch

from utils import Bar
from utils.evaluation.averagemeter import AverageMeter
from utils.evaluation.classification import performance
from utils.misc import (
    is_show,
    save_pred,
)
from utils.vizutils import viz_gt_pred
import pickle
def save_file(name2feature_seq, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(name2feature_seq, f)
# ----------------------------------------------------------------
# monkey patch for progress bar on SLURM
if True:
    #  disabling in interactive mode
    def writeln(self, line):
        on_slurm = os.environ.get("SLURM_JOB_ID", False)
        if self.file and (self.is_tty() or on_slurm):
            self.clearln()
            end = "\n" if on_slurm else ""
            print(line, end=end, file=self.file)
            self.file.flush()

    Bar.writeln = writeln
# ----------------------------------------------------------------


# Combined train/val epoch
def do_epoch(
    setname,
    loader,
    model,
    criterion,
    epochno=-1,
    optimizer=None,
    num_classes=None,
    debug=False,
    checkpoint=None,
    mean=torch.Tensor([0.5, 0.5, 0.5]),
    std=torch.Tensor([1.0, 1.0, 1.0]),
    feature_dim=1024,
    save_logits=False,
    save_features=False,
    num_figs=100,
    topk=[1],
    save_dir='',
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter()]
    perfs = []

    if save_features:
        all_features = torch.Tensor(loader.dataset.__len__(), feature_dim)

    model.eval()

    end = time.time()

    bar = Bar("E%d" % (epochno), max=len(loader))
    for i, data in enumerate(loader):
        print(i)
        if data.get("gpu_collater", False):
            # We handle collation on the GPU to enable faster data augmentation
            with torch.no_grad():
                data["rgb"] = data["rgb"].cuda()
                collater_kwargs = {}
                if isinstance(loader.dataset, torch.utils.data.ConcatDataset):
                    cat_datasets = loader.dataset.datasets
                    collater = cat_datasets[0].gpu_collater
                    cat_datasets = {
                        type(x).__name__.lower(): x for x in cat_datasets
                    }
                    collater_kwargs["concat_datasets"] = cat_datasets
                else:
                    collater = loader.dataset.gpu_collater
                data = collater(minibatch=data, **collater_kwargs)

        # measure data loading time
        data_time.update(time.time() - end)

        inputs = data["rgb"]

        inputs_cuda = inputs.cuda()


        # forward pass
        with torch.no_grad():
            outputs_cuda = model(inputs_cuda)

        # compute the loss



        if save_features:
            all_features[data["index"]] = outputs_cuda["embds"].squeeze().data.cpu()  # TODO


        # plot progress
        bar.suffix = "({batch}/{size}) Data: {data:.1f}s | Batch: {bt:.1f}s | Total: {total:} | ETA: {eta:} | ".format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
        )
        bar.next()
    bar.finish()

    num_clips = loader.dataset.num_clips
    st=0

    os.makedirs(save_dir, exist_ok=True)
    if save_features:
        for i in range(len(num_clips)):
            feature_single=all_features[st:st+num_clips[i]]
            st+=num_clips[i]
            entry = {'name': loader.dataset.train[i],
                     'feature': feature_single.numpy()}
            # name2feature_seq.append(entry)
            save_file(name2feature_seq=entry,
                      output_file=os.path.join(save_dir, '%s.pkl' % (loader.dataset.train[i].split('/')[-1].split('.mp4')[0])))
    return losses, perfs
