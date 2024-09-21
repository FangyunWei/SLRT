# Towards Online Continuous Sign Language Recognition and Translation
Code for EMNLP 2024 paper: Towards Online Continuous Sign Language Recognition and Translation


## Environment
It is better to use docker:
```
docker pull rzuo/pose:sing_ISLR
```
Or you may run 
```
pip install -r requirements.txt
```


## Data Preparation
### CSLR Datasets
We validate our method on [Phoenix-2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/), [Phoenix-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/), and [CSL-Daily](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/).
Note that all raw videos need to be zipped.

### Keypoints
The extracted keypoints are the same as those used in [TwoStream-SLR](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork).

### S3D K400 pretrained model
The S3D pretrained on K400 is avaialble [here](https://github.com/kylemin/S3D).
Put it in ``../pretrained_models/s3ds_actioncls_ckpt``


## Online CSLR Framework
Please check [CSLR/README.md](CSLR/README.md)


## Online SLT
Please check [SLT/README.md](SLT/README.md)


## Boosting an Offline Model with the Online Model
Please check [CTC_fusion/README.md](CTC_fusion/README.md)