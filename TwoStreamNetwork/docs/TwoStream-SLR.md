# TwoStream SLR

| Dataset | WER | Model | Training |
| :---: | :---: | :---: | :---: | 
| Phoenix-2014 | 18.8 | [ckpt]() | [config](../experiments/configs/TwoStream/phoenix-2014_s2g.yaml) |
| Phoenix-2014T | 19.3 | [ckpt]() | [config](../experiments/configs/TwoStream/phoenix-2014t_s2g.yaml) |
| CSL-Daily | 25.3 | [ckpt]() | [config](../experiments/configs/TwoStream/csl-daily_s2g.yaml) |

Here we describe the training of the two-stream network for sign language recognition task. 

## SingleStream Pretraining
To separately pretrain two visual encoders for videos and keypoints, run
```
dataset=phoenix-2014t # phoenix-2014t / phoenix-2014 / csl-daily
python -m torch.distributed.launch --nproc_per_node 8 --use_env training.py --config experiments/configs/TwoStream/${dataset}_video.yaml  #for videos
python -m torch.distributed.launch --nproc_per_node 8 --use_env training.py --config experiments/configs/TwoStream/${dataset}_keypoint.yaml  #for keypoints
```

## TwoStream Training
To load two pretrained encoders and train the dual visual encoder, run
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env training.py --config experiments/configs/TwoStream/${dataset}_s2g.yaml
```

## Evaluation
```
python -m torch.distributed.launch --nproc_per_node 1 --use_env prediction.py --config experiments/configs/TwoStream/${dataset}_s2g.yaml
```
## Checkpoints
We provide checkpoints trained by each stage [here].()