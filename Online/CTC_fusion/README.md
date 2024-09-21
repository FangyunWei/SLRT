# Boosting an Offline Model with the Online Model
Code for boosting an offline model, TwoStream-SLR, with the well-optimized online model. It also supports online inference of TwoStream-SLR.


## Data Preparation
Please check [../README.md](../README.md)


## Training
Training the fused model needs a well-optimized ISLR model, please check [../CSLR/README.md](../CSLR/README.md) in advance.
```
config_file='configs/phoenix-2014t_fuse_online.yaml'
python -m torch.distributed.launch --nproc_per_node 8 --master_port 29999 --use_env training.py --config=${config_file} 
```


## Testing (offline inference)
```
config_file='configs/phoenix-2014t_fuse_online.yaml'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29999 --use_env prediction.py --config=${config_file}
```


## Test Online TwoStream-SLR
```
config_file='configs/phoenix-2014t_s2g.yaml'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29999 --use_env prediction_online.py --config=${config_file}
```