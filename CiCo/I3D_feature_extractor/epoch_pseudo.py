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
import torch.nn.functional as F


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
    save_dir=''
):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter()]
    perfs = []
    for k in topk:
        perfs.append(AverageMeter())


    records={}

    model.eval()

    end = time.time()

    gt_win, pred_win, fig_gt_pred = None, None, None
    bar = Bar("E%d" % (epochno ), max=len(loader))
    for i, data in enumerate(loader):
        # print(i)
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
        with torch.no_grad():
        # forward pass
            outputs_cuda = model(inputs_cuda)


        # compute the loss

            after_soft_max = F.softmax(outputs_cuda["logits"],dim=-1)

            top_value, top_label = torch.max(after_soft_max, dim=-1)
            index = data['index']

            frame = data['frame']
            video_path = data['data_index']

        for iters in range(top_value.shape[0]):
            if top_value[iters] > 0.6:
                if top_label[iters].item() in records:
                    if video_path[iters] in records[top_label[iters].item()]:
                        records[top_label[iters].item()][video_path[iters]].append({'top_value': top_value[iters].item(), 'top_label': top_label[iters].item(),
                                         'frame': frame[iters], 'video_path': video_path[iters]})
                    else:
                        records[top_label[iters].item()][video_path[iters]]=[{'top_value': top_value[iters].item(), 'top_label': top_label[iters].item(),
                                         'frame': frame[iters], 'video_path': video_path[iters]}]
                else:
                    records[top_label[iters].item()]={}
                    records[top_label[iters].item()][video_path[iters]] = [
                        {'top_value': top_value[iters].item(), 'top_label': top_label[iters].item(),
                         'frame': frame[iters], 'video_path': video_path[iters]}]

        # plot progress
        # if (1+1)%100==0:
        bar.suffix = "({batch}/{size}) Data: {data:.1f}s | Batch: {bt:.1f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:} | Perf: {perf:}".format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=", ".join([f"{losses[i].avg:.3f}" for i in range(len(losses))]),
            perf=", ".join([f"{perfs[i].avg:.3f}" for i in range(len(perfs))]),
        )
        bar.next()
    bar.finish()

    num_clips = loader.dataset.num_clips
    st=0

    os.makedirs(save_dir, exist_ok=True)
    print("save_dir",save_dir)

    for words in records:
        for sim_word_in_one_video in records[words]:
            curr_list=records[words][sim_word_in_one_video]
            curr_list=sorted(curr_list, key=lambda x: x['top_value'], reverse=True)
            frames=curr_list[0]['frame']
            frame_start=np.min(frames)
            record_dict={}
            record_score={}
            record_dict[frame_start]=frames
            record_score[frame_start]=curr_list[0]['top_value']
            for iterations in range(len(curr_list)):
                if iterations==0:
                    continue
                curr_frame = curr_list[iterations]['frame']
                curr_start_frame=np.min(curr_frame)
                isin=0
                for frame_start_key in record_dict:
                    if np.abs(curr_start_frame-frame_start_key)<=3:
                        record_dict[frame_start]+=curr_frame
                        isin=1
                    elif np.abs(curr_start_frame-frame_start_key)<=24:
                        isin = 1

                if isin==0:
                    record_dict[curr_start_frame] = curr_frame
                    record_score[curr_start_frame]=curr_list[iterations]['top_value']
            for items in record_dict:
                top_value=record_score[items]
                top_label=words
                frame=record_dict[items]
                video_path=sim_word_in_one_video

                video_file_name=video_path.split('/')[-1].split('.')[0]
                store_name=video_file_name+'_stframes_'+str(np.min(frame))+'_endframes_'+str(np.max(frame))+'_logits_'+str(np.round(top_value,3))
                store_dir = f'{save_dir}/'+ '%04d' % (top_label)
                os.makedirs(store_dir,exist_ok=True)
                save_video_seg(video_path,np.min(frame),np.max(frame),os.path.join(store_dir,store_name))

import cv2
import  PIL.Image as Image
def save_video_seg(video_file,start,end,store_name):

    if not os.path.isfile(video_file):
        return None
    cap = cv2.VideoCapture(video_file)
    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if end>frames:
        start=max(0,frames-16)
        end=frames
    imgs = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        # Avoid problems when video finish
        if ret==True:
            # Croping the frame
            crop_frame = frame
            crop_frame_ = Image.fromarray(crop_frame)
            #crop_frame_ = cv2.cvtColor(np.asarray(crop_frame_), cv2.COLOR_RGB2BGR)
            imgs.append(crop_frame_)
        else:
            break
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(store_name+'.mp4', fourcc, fps,(w_frame, h_frame))
    try:
        for img_ in imgs[start:end]:
            out.write(np.asarray(img_))
    except:
        print(start,end)
    out.release()
