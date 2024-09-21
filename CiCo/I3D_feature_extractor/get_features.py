import os
import argparse


import os
import argparse

parser = argparse.ArgumentParser(description="AML Generic Launcher")
parser.add_argument('--gpuid', default="0",type=str, help="Which GPU to use")

args, _ = parser.parse_known_args()
for i in range(256):
        command_str="  python extract_sign_features.py --rank %d --gpuid %s  --root_path  ../data/How2Sign/processed_videos --pretrained chpt/bsl5k.pth.tar --save_dir ../CLCL/sign_features/h2s_domain_agnostic --split train"%(i,args.gpuid)
        print(command_str)
        os.system(command_str)

for i in range(256):
        command_str="  python extract_sign_features.py --rank %d --gpuid %s  --root_path  ../data/How2Sign/processed_videos --pretrained chpt/domain_aware_I3D_H2S.pth.tar --save_dir ../CLCL/sign_features/h2s_domain_aware --split train"%(i,args.gpuid)
        print(command_str)
        os.system(command_str)


for i in range(16):
    command_str = "  python extract_sign_features.py --rank %d --gpuid %s  --root_path  ../data/How2Sign/processed_videos --pretrained chpt/bsl5k.pth.tar --save_dir ../CLCL/sign_features/h2s_domain_agnostic --split test" % (i, args.gpuid)
    print(command_str)
    os.system(command_str)

for i in range(16):
        command_str="  python extract_sign_features.py --rank %d --gpuid %s  --root_path  ../data/How2Sign/processed_videos --pretrained chpt/domain_aware_I3D_H2S.pth.tar --save_dir ../CLCL/sign_features/h2s_domain_aware --split test"%(i,args.gpuid)
        print(command_str)
        os.system(command_str)