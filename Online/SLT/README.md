# Online Sign Language Translation
Code for online sign language translation with a wait-k gloss2text network.
The implementation of wait-k policy (transformers_cust) is mainly based on the [repo](https://github.com/prajdabre/yanmtt).


## Data Preparation
Please first collect predicted glosses of the online SLR model. See Online Testing in [../CSLR/README.md](../CSLR/README.md) and [../CTC_fusion/README.md](../CTC_fusion/README.md) for more details.


## Training
```
config_file='configs/g2t_wait2.yaml'
python -m torch.distributed.launch --nproc_per_node 8 --master_port 29999 --use_env training.py --config=${config_file} 
```

## Testing
```
config_file='configs/g2t_wait2.yaml'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29999 --use_env prediction.py --config=${config_file}
```