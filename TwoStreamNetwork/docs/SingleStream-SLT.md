# A Simple Multi-modality Transfer Learning Baseline for Sign Language Translation (SingleStream-SLT Baseline)

| Dataset | R | B1 | B2 | B3 | B4 | Model | Training  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Phoenix-2014T | 53.08 | 54.48 | 41.93 | 33.97 | 28.57 | [ckpt](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EkpQzXWBrWxDruz6-U-V0kUBgit2vXsc40wLipV8YPLXaQ?e=Bru3oz) | [config](../experiments/configs/SingleStream/phoenix-2014t_s2t.yaml) |
| CSL-Daily | 53.35 | 53.53 | 40.68 | 31.04 | 24.09 |[ckpt](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EudFBd-IzWJOngYdXVxox6kBI7ASATileGu8ncW-dBDi-w?e=YvKAKm) | [config](../experiments/configs/SingleStream/csl-daily_s2t.yaml) |


## Pretraining

The general-domain pretraining is already done by loading the pretrained checkpoints, i.e. S3D and mBart. We then apply Sign2Gloss and Gloss2Text within-domain pretraining for the visual and language module respectively. 

For Sign2Gloss pretraining, run
```
dataset=phoenix-2014t #phoenix-2014t / csl-daily
python -m torch.distributed.launch --nproc_per_node 8  --use_env training.py --config experiments/configs/SingleStream/${dataset}_s2g.yaml 
```

For Gloss2Text pretraining, run
```
python -m torch.distributed.launch --nproc_per_node 8  --use_env training.py --config experiments/configs/SingleStream/${dataset}_g2t.yaml
```

## Multi-modal Joint Training

First, to extract features output by the pretrained S3D, run
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env extract_feature.py --config experiments/configs/SingleStream/${dataset}_s2g.yaml
```
We provide our pre-extracted features for [Phoenix-2014T](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EndgQUATcNRCj0pTKPNMA_kBxSE9iJSONqj1zq1kQAAn5g?e=BgbJCK) and [CSL-Daily](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EjbL5fTAZbxOmGA5x7px8s8BbyJ4ml5e5TROB-GEWPXeBQ?e=Ks7GfH).

For multi-modal joint training, run

```
python -m torch.distributed.launch --nproc_per_node 1 --use_env training.py --config experiments/configs/SingleStream/${dataset}_s2t.yaml
```

## Evaluation 

To evaluate Sign2Text performance, run
```
python -m torch.distributed.launch --nproc_per_node 1 --use_env prediction.py --config experiments/configs/SingleStream/${dataset}_s2t.yaml
```
## Checkpoints
We provide checkpoints trained by each stage [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EuJlnAhX7h9NnvFZhQH-_fcBtV8lbnj2CphiuidhhcU69w?e=eOsQ4B).
