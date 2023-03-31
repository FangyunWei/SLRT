# Example usage:
"""
source activate pytorch1.3
python evaluate_seq.py --datasetname phoenix2014 \
    --checkpoint checkpoint/phoenix2014/T_c1233_ctc_blank_unfreeze/test_002_stride0.50/ \
    --num-classes 1233 --num_in_frames 16 --stride 0.5 \
    --phoenix_path data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
"""
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import datasets
import opts
from utils.clean_phoenix import clean_phoenix_2014, clean_phoenix_2014_trans
from utils.evaluation.wer import wer_list, wer_single

plt.switch_backend("agg")


def phoenix_make_ctm(pred, names, output_file="demo.ctm"):
    f = open(output_file, "w")
    interval = 0.1
    for i in range(len(pred)):
        t_beg = 0
        for gloss in pred[i]:
            f.write(f"{names[i]} {t_beg:.3f} {t_beg + interval:.3f} {gloss}\n")
            t_beg += interval
    f.close()


def phoenix_name_to_official(name, phoenix2014T=False):
    name = name[:-4]  # rm ".mp4"
    name = name.split("/")[-1]  # rm "test/"
    if phoenix2014T:
        return f"{name} 1"
    else:
        sp = name.split("_")
        return f"{name[:-2]} {sp[-1]}"


def gather_clips(dataloader_val, scores=None, features=None, phoenix2014T=False):
    """
        Takes in sliding window network outputs (clips)
            dataloader_val: dataloader constructed with evaluate_video=1
            scores: clip scores [num_clips, num_classes]
            features: clip features [num_clips, feature_dim]
    """
    video_ix = np.unique(dataloader_val.valid)
    N = len(video_ix)
    gt = [None for _ in range(N)]
    pred = [None for _ in range(N)]
    gt_glosses = [None for _ in range(N)]
    pred_glosses = [None for _ in range(N)]
    names = [None for _ in range(N)]
    len_clip = np.zeros(N)
    if features is not None:
        vid_features = np.zeros((N, features.shape[1]))
    else:
        vid_features = None

    for i, vid in enumerate(video_ix):
        clip_ix = np.where(dataloader_val.valid == vid)
        if scores is not None:
            clip_score = scores[clip_ix]
            len_clip[i] = clip_score.shape[0]
            pred_seq = np.argmax(clip_score, axis=1)
            # [8, 8, 59, 603, 603, 603, 8] becomes [8, 59, 603, 8] removing consecutive repeating elems
            pred_seq_simple = [
                v for i, v in enumerate(pred_seq) if i == 0 or v != pred_seq[i - 1]
            ]
            pred[i] = pred_seq_simple
            gt[i] = dataloader_val.classes[vid]
            pred_glosses[i] = [
                "".join(dataloader_val.class_names[g].split(" ")[1:]) for g in pred[i]
            ]
            gt_glosses[i] = [
                "".join(dataloader_val.class_names[g].split(" ")[1:]) for g in gt[i]
            ]
            names[i] = phoenix_name_to_official(dataloader_val.videos[vid], phoenix2014T)
        if features is not None:
            vid_features[i] = np.mean(features[clip_ix], axis=0)

    return gt, pred, gt_glosses, pred_glosses, names, len_clip, vid_features


def get_dataloader(args):
    common_kwargs = {
        "stride": args.stride,
        "inp_res": args.inp_res,
        "resize_res": args.resize_res,
        "num_in_frames": args.num_in_frames,
    }
    if args.datasetname == "phoenix2014":
        loader = datasets.PHOENIX2014(
            root_path=args.phoenix_path,
            setname=args.test_set,
            evaluate_video=args.evaluate_video,
            **common_kwargs,
        )
    else:
        print("Which dataset? (evaluate.py)")
        exit()
    return loader


def save_wer_to_json(wer_result, result_file):
    print(f"Saving results to {result_file}")
    with open(result_file, "w") as f:
        f.write(json.dumps(wer_result, indent=2))
        f.write("\n")


def evaluate(args, plog):
    if "PHOENIX-2014-T" in args.phoenix_path:
        phoenix2014T = True
    with_scores = True
    with_features = args.save_features
    dataloader_val = get_dataloader(args)
    exp_root = args.checkpoint
    scores = None
    if with_scores:
        scores_file = f"{exp_root}/preds_valid.mat"
        plog.info(f"Loading from {scores_file}")
        scores = sio.loadmat(scores_file)["preds"]
        plog.info(scores.shape)  # e.g. [32558, 60]
        assert scores.shape[0] == len(dataloader_val.valid)
    features = None
    if with_features:
        plog.info("Loading features_valid.mat")
        features_file = f"{exp_root}/features_valid.mat"
        features = sio.loadmat(features_file)["preds"]

    # Aggregate the clips of each video, compute the GT/Pred/Feature for each video.
    gt, pred, gt_glosses, pred_glosses, names, len_clip, vid_features = gather_clips(
        dataloader_val, scores=scores, features=features, phoenix2014T=phoenix2014T,
    )

    N = len(gt)
    gt_sentences = [" ".join(s) for s in gt_glosses]
    pred_sentences = [" ".join(s) for s in pred_glosses]
    gt_list = []
    pred_list = []
    for i in range(N):
        if phoenix2014T:
            gt_i_clean = clean_phoenix_2014_trans(gt_sentences[i])
            pred_i_clean = clean_phoenix_2014_trans(pred_sentences[i])
        else:
            gt_i_clean = clean_phoenix_2014(gt_sentences[i])
            pred_i_clean = clean_phoenix_2014(pred_sentences[i])
        w = wer_single(gt_i_clean, pred_i_clean)
        gt_list.append(gt_i_clean)
        pred_list.append(pred_i_clean)
        print("GT", gt_i_clean)
        print("PD", pred_i_clean)
        print(w)
        print()
    wer_result = wer_list(gt_list, pred_list)
    for k, v in wer_result.items():
        wer_result[k] = f"{v:.2f}"
    print("==> Python eval script results:")
    print(wer_result)
    if phoenix2014T:
        eval_path = "/users/gul/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/evaluation/sign-recognition/"
        eval_script = "evaluatePHOENIX-2014-T-signrecognition.sh"
    else:
        eval_path = "/users/gul/datasets/phoenix2014-release/phoenix-2014-multisigner/evaluation/"
        eval_script = "evaluatePhoenix2014_gul.sh"

    try:
        # Try evaluating with official script
        output_file = "demo.ctm"
        phoenix_make_ctm(pred_glosses, names, output_file=f"{eval_path}/{output_file}")
        print("==> Official eval script results:")
        cmd = f"cd {eval_path} && PATH=$PATH:/users/gul/tools/sctk-2.4.10/bin ./{eval_script} {output_file} test"
        # out = os.system(cmd)
        out = os.popen(cmd).read()
        print(out)
        wer_official = float(out[out.index("=") + 1:out.index("%")].strip())
        print(wer_official)
        wer_result["wer_official"] = wer_official
    except:
        print("Official evaluation broke for some reason.")

    # Write the results to json file
    result_file = f"{exp_root}/wer.json"
    save_wer_to_json(wer_result, result_file)
    return wer_result["wer"]


if __name__ == "__main__":
    args = opts.parse_opts()
    args.evaluate_video = 1
    args.test_set = "test"
    plog = logging.getLogger("eval")
    evaluate(args, plog)
