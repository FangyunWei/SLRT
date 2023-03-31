import argparse
import collections
import datetime
import json
import os
import pickle
from pathlib import Path

from mergedeep import Strategy, merge
from beartype import beartype
from zsvision.zs_utils import set_nested_key_val, load_json_config

import models

model_names = sorted(
    name
    for name in models.__dict__
    # if name.islower() and not name.startswith("__")
    if not name.startswith("__")
    and isinstance(models.__dict__[name], collections.Callable)
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    # Model structure
    parser.add_argument(
        "--arch",
        "-a",
        default="InceptionI3d",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names),
    )
    parser.add_argument(
        "--num-classes", default=5383, type=int, metavar="N", help="Number of classes"
    )
    # Training strategy
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--gpuid",
        default='1',
        type=str,
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--train-batch", default=18, type=int, metavar="N", help="train batchsize"
    )
    parser.add_argument(
        "--test-batch", default=18, type=int, metavar="N", help="test batchsize"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-2,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=0,
        type=float,
        metavar="W",
        help="weight decay (default: 0)",
    )
    parser.add_argument(
        "--schedule",
        type=int,
        nargs="*",
        default=[20, 40],
        help="Decrease learning rate at these epochs.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="LR is multiplied by gamma on schedule.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="LR is multiplied by gamma on schedule.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='try',
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="LR is multiplied by gamma on schedule.",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=512,
        help="LR is multiplied by gamma on schedule.",
    )


    # Miscs
    parser.add_argument(
        "--snapshot", default=5, type=int, metavar="N", help="frequency of saving model"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoint",
        type=str,
        metavar="PATH",
        help="path to save checkpoint (default: checkpoint)",
    )

    parser.add_argument(
        "--resume",
        default="chpt/bsl5k.pth.tar",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        default=True,
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        help="show intermediate results",
    )
    parser.add_argument(
        "--inp_res",
        type=int,
        default=224,
        help="Spatial resolution of the network input.",
    )
    parser.add_argument(
        "--resize_res",
        type=int,
        default=256,
        help="Spatial resolution of the resized input before crop (300 | 130).",
    )
    parser.add_argument(
        "--num_in_frames", type=int, default=16, help="Number of input frames."
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1024,
        help="Dimensionality of the feature before classification layer",
    )
    parser.add_argument(
        "--save_features", type=int, default=1, help="Whether to save features."
    )
    parser.add_argument(
        "--pretrained", type=str, default="", help="path to pretrained model file"
    )
    parser.add_argument(
        "--evaluate_video",
        type=int,
        default=1,
        help="whether to test on sliding windows",
    )
    parser.add_argument(
        "--stride", type=float, default=1, help="stride ratio of sliding windows"
    )
    parser.add_argument(
        "--test_set",
        type=str,
        default="val",
        help="Which set to evaluate on: val | test",
    )
    parser.add_argument(
        "--nloss", type=int, default=1, help="number of losses to keep track of"
    )
    parser.add_argument(
        "--nperf", type=int, default=2, help="number of performance metrics"
    )
    parser.add_argument(
        "--num_figs",
        type=int,
        default=10,
        help="frequency to save figures (default 10)",
    )
    parser.add_argument(
        "--init_cross_language",
        type=str,
        default="",
        help=("Whether to do hacks to initialize classifier weights with corresponding "
              "asl/bsl signs. Options: asl_with_bsl, bsl_with_asl"),
    )
    parser.add_argument(
        "--asl_dataset",
        type=str,
        default=None,
        choices=[None, "wlasl", "msasl"],
        help=("if set, this defines the asl dataset used to perform the cross language"
              " mapping"),
    )
    parser.add_argument(
        "--gpu_collation",
        type=int,
        default=256,
        help="If set, shift the collation and preprocessing onto the GPU.",
    )
    parser.add_argument(
        "--ram_data",
        type=int,
        default=0,
        help="If set, enable the use of in-memory datasets (pickled video frames).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=[1, 5],
        help="A list of k values for which to compute top-k accuracies.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="The number of gpus to use for training and testing",
    )
    parser.add_argument(
        "--bsl1k_mouthing_prob_thres",
        type=float,
        default=0.5,
        help="The mouthing score for bsl1k training.",
    )
    parser.add_argument(
        "--bsl1k_num_last_frames",
        type=int,
        default=20,
        help="Number of frames to train with before the mouthing peak",
    )
    parser.add_argument("--datasetname", type=str, default='H2S',help="dataset")
    parser.add_argument(
        "--featurize_mode",
        type=int,
        default=1,
        help="run a single epoch of feature extraction over each subset",
    )
    parser.add_argument(
        "--featurize_mask",
        default="",
        help="only featurize videos that pass this string filter",
    )
    parser.add_argument(
        "--word_data_pkl", type=str, default="misc/bsl1k/bsl1k_vocab.pkl", help="Path to the list of words."
    )
    parser.add_argument(
        "--phoenix_assign_labels", type=str, default="auto", help="uniform | auto"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default=None,
        help="The path to Phoenix2014T dataset",
    )
    parser.add_argument(
        "--bsl1k_pose_subset",
        type=int,
        default=0,
        help="Use the subset of the bsl1k dataset that has extracted pose.",
    )
    parser.add_argument(
        "--input_type", type=str, default="rgb", help="Options: rgb | pose"
    )
    parser.add_argument(
        "--pose_keys",
        type=str,
        nargs="+",
        default=["body", "face", "lhnd", "rhnd"],
        help="List of body parts to use (to be used with Pose2Sign architecture)",
    )
    parser.add_argument(
        "--mask_rgb",
        type=str,
        default=None,
        help="Applies to bsl1k/msasl/wlasl. Options: face | mouth",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default=None,
        help="Applies to bsl1k/msasl/wlasl. Options: include | exclude",
    )
    parser.add_argument(
        "--bsl1k_anno_key",
        type=str,
        default="original-mouthings",
        help="Whether to train with pseudo annotations",
    )
    parser.add_argument(
        "--include_embds",
        type=int,
        default=1,
        help="Whether to return the I3D embeddings.",
    )
    return parser


def parse_opts(argv=None):
    parser = build_parser()
    return parser.parse_args(argv)


def print_args(args):
    print("==== Options ====")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("=================")


def save_args(args, save_folder, opt_prefix="opt", verbose=True):
    opts = vars(args)
    os.makedirs(save_folder, exist_ok=True)

    # Save to text
    opt_filename = f"{opt_prefix}.txt"
    opt_path = os.path.join(save_folder, opt_filename)
    with open(opt_path, "a") as opt_file:
        opt_file.write("====== Options ======\n")
        for k, v in sorted(opts.items()):
            opt_file.write(f"{str(k)}: {str(v)}\n")
        opt_file.write("=====================\n")
        opt_file.write(f"launched at {str(datetime.datetime.now())}\n")

    # Save as pickle
    opt_picklename = f"{opt_prefix}.pkl"
    opt_picklepath = os.path.join(save_folder, opt_picklename)
    with open(opt_picklepath, "wb") as opt_file:
        pickle.dump(opts, opt_file)
    if verbose:
        print(f"Saved options to {opt_path}")
