# NLA-SLR
Official Implementations for [Natual Language-Assisted Sign Language Recognition, CVPR 2023]()

## Introduction
<img src="images/similar.png" width="400"><img src="images/distinct.png" width="400">
<img src="images/idea.png" width="800">

There exists a lot of visually indistinguishable signs (VISigns) in current SLR datasets. They can be categorized into two classes: VISigns with similar or distinct semantic meanings. However, purely vision-based neural networks are less effective to recognize these VISigns. To address this issue, we propose the Natural Language-Assisted Sign Language Recognition (NLA-SLR) framework, which exploits semantic information contained in glosses (sign labels). First, for VISigns with similar semantic meanings, we propose language-aware label smoothing by generating soft labels for each training sign whose smoothing weights are computed from the normalized semantic similarities among the glosses to ease training. Second, for VISigns with distinct semantic meanings, we present an inter-modality mixup technique which blends vision and gloss features to further maximize the separability of different signs under the supervision of blended labels.

## Performance and Checkpoints

| Dataset | P-I Top1 | P-I Top5 | P-C Top1 | P-C Top5 | Ckpt | Training |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| WLASL-2000 | 61.26 | 91.77 | 58.31 | 90.91 | [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EhR1iEfa7MpGuQ4hnKiPN4wB-NwWEfDH0dzZUdu4eUYOwg?e=ghtdJU)/[Baidu](https://pan.baidu.com/s/16Dl82PJFThv6McITf8MeWQ?pwd=brcz) | [config](configs/nla_slr_wlasl_2000.yaml) |
| WLASL-1000 | 75.64 | 94.62 | 75.72 | 94.65 | [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EkxYtFaTXzdPveWTsmW_2GoB0pJVLkHbSbfLP4LIWb6Olg?e=hk0neS)/[Baidu](https://pan.baidu.com/s/1IY5aPrab0ZtHFfu4YFKl7Q?pwd=fi1s) | [config](configs/nla_slr_wlasl_1000.yaml) |
| WLASL-300 | 86.98 | 97.60 | 87.33 | 97.81 | [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EiACoQ3dgP5IuXFmY74wnPABKHA2cuojpiqQdTG-ToB5xg?e=liV5l9)/[Baidu](https://pan.baidu.com/s/1YRtsLyxb11vpvZSHeln57Q?pwd=xb5c) | [config](configs/nla_slr_wlasl_300.yaml) |
| WLASL-100 | 92.64 | 96.90 | 93.08 | 97.17 | [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/ErxKSls_d_BAlbCPsN3AOWcBNnHQ1YydTrdJzwaDKZBYkg?e=DRVtZr)/[Baidu](https://pan.baidu.com/s/1h9omKOSJYvcwG_0AWudnLQ?pwd=u8p3) | [config](configs/nla_slr_wlasl_100.yaml) |
| MSASL-1000 | 73.80 | 89.65 | 70.95 | 89.07 | [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/Egss6sJoojlOul47Q2Mf5Y8B-gnbhIu-Lo9PoUoclER6Tg?e=ukn9Bu)/[Baidu](https://pan.baidu.com/s/1YbOmqablXwydvv_IRQ3Fbg?pwd=r4ni) | [config](configs/nla_slr_msasl_1000.yaml) |
| MSASL-500 | 82.90 | 93.46 | 83.06 | 93.54 | [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/Elllx63j_ptOozN6uHt2IIkBVKmPvw5oUeARWXyndYCA3Q?e=aoVP22)/[Baidu](https://pan.baidu.com/s/19wh7B5AhhYRc8QP7fZy3oQ?pwd=mw9m) | [config](configs/nla_slr_msasl_500.yaml) |
| MSASL-200 | 89.48 | 96.69 | 89.86 | 96.93 | [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/Eqega94so2JBrwtU8H7Cyi4BfarxXh0U6GEkWo9E5ci6oA?e=T3pxyI)/[Baidu](https://pan.baidu.com/s/1s0HrP3RuBcQn2ckgsGN7EA?pwd=x3r7) | [config](configs/nla_slr_msasl_200.yaml) |
| MSASL-100 | 91.02 | 97.89 | 91.24 | 98.19 | [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EuR4Cadp8XJEmBKJy_6S1gEBiG2YV7JE9LfRDxWYXFIakg?e=bg9ptD)/[Baidu](https://pan.baidu.com/s/1eUqY5vDBKQr14gYqlm-wtQ?pwd=6us1) | [config](configs/nla_slr_msasl_100.yaml) |
| NMFs-CSL | 83.7 | 98.5 | -- | -- | [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/Eg_SSR-UYnlFm1RzN_AWtIwBS4s3rd5Awb3e4fMQwXsGSw?e=XymJNe)/[Baidu](https://pan.baidu.com/s/1qhQLPlQGW9zZkOf-OkB7LQ?pwd=mv1g) | [config](configs/nla_slr_nmf.yaml) |

More checkpoints, e.g., single streams and VKNet-64/32, are available at [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/ErKHcJhsiAdKgkm2djnFAcoBJt4NC0UG3TtvhJ9EAjfc5Q?e=3wvFh1)/[Baidu](https://pan.baidu.com/s/1dkbs0JqWukXMoY6gwV-XLw?pwd=up4z).

## Usage
### Environment
It is better to use docker:
```
docker pull rzuo/pose:sing_ISLR
```
You may also install packages by:
```
pip install -r requirements.txt
```

### Data Preparation
**SLR Datasets**

We use three datasets: [WLASL](https://dxli94.github.io/WLASL/), [MSASL](https://www.microsoft.com/en-us/research/project/ms-asl/), and [NMFs-CSL](http://home.ustc.edu.cn/~alexhu/Sources/index.html). Note that all raw videos need to be zipped.

**Word Embeddings**

We use fastText trained on Common Crawl. The pretrained word embeddings can be downloaded from [here](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip).

**Keypoints**

We use [HRNet](https://github.com/open-mmlab/mmpose/tree/master/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py) pretrained on COCO-WholeBody. Below is an example of extracting keypoints for the WLASL training set.
```
config_file='configs/nla_slr.yaml'
python -m torch.distributed.launch --nproc_per_node 8 --master_port 29999 --use_env gen_pose.py --config=${config_file} --split=train
```

**S3D K400 pretrained model**

Download S3D pretrained on K400 from [here](https://github.com/kylemin/S3D).
Put it in `../../pretrained_models/s3ds_actioncls_ckpt`

Split files, word embeddings, and keypoints are available at: WLASL ([OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EscNEp4RnyZOtJ-HuF8Qm9EB4KGBF72J2GjhwKdrxmN1Kg?e=VgAS8v)/[Baidu](https://pan.baidu.com/s/1G85Yum_SMjpWm3SlH_xBDw?pwd=9ry8)), MSASL ([OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EqKd-5xbWLZOqAS-v_4yilEBivjPL2-OCaJZXD6e_BR2Xg?e=cQSHwm)/[Baidu](https://pan.baidu.com/s/1xoUoqG3rUNHM_z3elLcVoQ?pwd=mvhw)), NMFs-CSL ([OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/Eu6D8qSnTFxMut3JYeBFyDIBW8MvKsh6jeyLH1mHfRmvnw?e=TaeJxr)/[Baidu](https://pan.baidu.com/s/10FGUodoygZkzW5G1QDKyAw?pwd=w4ro)).


### Training

Video/keypoint encoders (64-frame as an example):
```
config_file='configs/rgb_frame64.yaml'
python -m torch.distributed.launch --nproc_per_node 8 --master_port 29999 --use_env training.py --config ${config_file} 
```
```
config_file='configs/pose_frame64.yaml'
python -m torch.distributed.launch --nproc_per_node 8 --master_port 29999 --use_env training.py --config ${config_file} 
```

VKNet-64/32 (load pretrained video/keypoint encoders)
```
config_file='configs/two_frame64.yaml'
python -m torch.distributed.launch --nproc_per_node 8 --master_port 29999 --use_env training.py --config ${config_file} 
```
```
config_file='configs/two_frame32.yaml'
python -m torch.distributed.launch --nproc_per_node 8 --master_port 29999 --use_env training.py --config ${config_file} 
```

NLA-SLR (load VKNet-64/32)
```
config_file='configs/nla_slr_wlasl_2000.yaml'
python -m torch.distributed.launch --nproc_per_node 8 --master_port 29999 --use_env training.py --config ${config_file} 
```


### Testing
```
config_file='configs/nla_slr_wlasl_2000.yaml'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29999 --use_env prediction.py --config ${config_file} --eval_setting origin
```
3-crop inference can usually yield better results:
```
config_file='configs/nla_slr_wlasl_2000.yaml'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29999 --use_env prediction.py --config ${config_file} --eval_setting 3x_pad
```

## Citations
```
@inproceedings{zuo2023natural,
  title={Natural Language-Assisted Sign Language Recognition},
  author={Zuo, Ronglai and Wei, Fangyun and Mak, Brian},
  booktitle={CVPR},
  year={2023}
}
```