# TwoStream-SLT

| Dataset | R | B1 | B2 | B3 | B4 | Model | Training |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Phoenix-2014T | 53.48 | 54.90 | 42.43 | 34.46 | 28.95 | [video](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EphztyWWWudGjNoPugO53MYBNuv7FUATs1gpUufdtgrAow?e=J28eLg)/[keypoint](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/Eq34FYe33qlKpxWGM089rq8BFDM_hkd7b8ewgpg1RTpb9Q?e=dVw8LZ)/[joint](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/Et0ZNVTztKFEqpbOjotlfx4BtiIykhw27U6zQ3LQAJiRkQ?e=sgpB1q) | [config](../experiments/configs/TwoStream/phoenix-2014t_s2t_ensemble.yaml) |
| CSL-Daily | 55.72 | 55.44 | 42.59 | 32.87 | 25.79 | [video](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EmSUuTojKAZIpy90aA75s00BBOrlZyhkvFBNsbibtgx5mg?e=0MPPEn)/[keypoint](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EuZpa5hRV6tMvRFWngg86VUBi01T5GpQ5fkIfKHh571dbw?e=HRTaEG)/[joint](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EvAcRN1wDg5JmwdcojaGICMByzgNgq7CJFOqVTXQgV8Rrg?e=46Em1S) | [config](../experiments/configs/TwoStream/csl-daily_s2t_ensemble.yaml) |

We use the trained TwoStream-SLR for sign language translation (SLT) (See [TwoStream-SLR.md](TwoStream-SLR.md)).  The training procedure is similar to our first proposed SLT baseline (See [SingleStream-SLT.md](TwoStream-SLR.md)) except that we replace the SingleStream-SLR with a TwoStream-SLR, append three translation networks for the three SLR heads, and employ multi-source ensemble for inference. 
  
## Pretraining
Sign2Gloss pretraining is already done as described in [TwoStream-SLR.md](TwoStream-SLR.md).
For Gloss2Text pretraining for the language module (same as in SingleStream-SLT), run
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env training.py --config experiments/configs/SingleStream/${dataset}_g2t.yaml
```

## Multi-stream Multi-modal Joint training
First, to compute features output by the TwoStream visual encoder, run
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env extract_feature.py --config experiments/configs/TwoStream/${dataset}_s2g.yaml
```
We provide features extracted by our trained checkpoints for [Phoenix-2014T](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/Eu5cV-7VnZNAgGnSBLHh2b0BnJkSyLagpfDeIjSX-GXqjw?e=Ztxk3d) and [CSL-Daily](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EqwZKuG0gHJKpfIW1bjomT8BUxEfoh4P0wrWMba8L9Vn0w?e=C7AX2Y).

We seperately train three Sign2Text networks using outputs of the video head, keypoint head and joint head respectively.
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env training.py --config experiments/configs/TwoStream/${dataset}_s2t_video.yaml
python -m torch.distributed.launch --nproc_per_node 8 --use_env training.py --config experiments/configs/TwoStream/${dataset}_s2t_keypoint.yaml
python -m torch.distributed.launch --nproc_per_node 8 --use_env training.py --config experiments/configs/TwoStream/${dataset}_s2t_joint.yaml
``` 

## Evaluation (Multi-source ensemble
To ensemble predictions of the three translation networks, run
```
for stream in video keypoint joint
do
python -m torch.distributed.launch --nproc_per_node 1 --use_env extract_feature.py --config experiments/configs/TwoStream/${dataset}_s2t_${stream}.yaml --output_split dev,test
done

python -m torch.distributed.launch --nproc_per_node 1 --use_env prediction.py --config experiments/configs/TwoStream/${dataset}_s2t_ensemble.yaml
```

## Checkpoints
We provide checkpoints trained by each stage [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EpyVs_YNq2NKrxn0FPJJWF4BtS7O1wTrOEa2ZvMwT2OU-g?e=u0sQab).
