import os
import argparse


import os
import argparse

parser = argparse.ArgumentParser(description="AML Generic Launcher")
parser.add_argument('--gpuid', default="0",type=str, help="Which GPU to use")

args, _ = parser.parse_known_args()
for i in range(256):
        command_str="  python  generate_pseudo_labels.py --rank %d --gpuid %s --root_path ../data/How2Sign/processed_videos --pretrained chpt/bsl5k.pth.tar --save_dir ../I3D_trainer/pseudo_h2s --split train"%(i,args.gpuid)
        print(command_str)
        os.system(command_str)

for i in range(16):
        command_str="  python  generate_pseudo_labels.py --rank %d --gpuid %s --root_path ../data/How2Sign/processed_videos --pretrained chpt/bsl5k.pth.tar --save_dir ../I3D_trainer/pseudo_h2s --split test"%(i,args.gpuid)
        print(command_str)
        os.system(command_str)