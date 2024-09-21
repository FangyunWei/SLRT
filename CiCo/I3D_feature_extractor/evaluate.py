# Example usage:
# source activate pytorch1.3_env
""" python evaluate.py \
    --datasetname bsl1k \
    --checkpoint checkpoint/bsl1k/c1064_16f_unfreezei3d_m9/test_050/ \
    --num-classes 1064 --num_in_frames 16 --stride 0.5 \
    --nperf 3 --topk 1 5 10 \
    --word_data_pkl misc/bsldict/subtitles/data/words_mouthing0.8_1064_20.02.21.pkl \
    --bsl1k_num_last_frames 20 \

python evaluate.py \
    --datasetname bsl1k \
    --checkpoint data/checkpoint/bsl1k/c1064_16f_unfreezei3d_m5_last20_poseinit_cls_balanced_sampling/2020-06-29_07-48-03/test_050 \
    --num-classes 1064 --num_in_frames 16 --stride 0.5 \
    --nperf 3 --topk 1 5 10 \
    --word_data_pkl misc/bsldict/subtitles/data/words_mouthing0.8_1064_20.02.21.pkl \
    --bsl1k_num_last_frames 20
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import scipy.io as sio

import opts
import datasets

plt.switch_backend("agg")


def aggregate_clips(dataloader_val, topk=[1], scores=None, features=None):
    """
        Takes in sliding window network outputs (clips), averages over the videos
            dataloader_val: dataloader constructed with evaluate_video=1
            topk: list of k values
            scores: clip scores [num_clips, num_classes]
            features: clip features [num_clips, feature_dim]
    """
    video_ix = np.unique(dataloader_val.valid)
    N = len(video_ix)
    maxk = max(topk)
    gt = np.zeros(N)
    pred = np.zeros((N, maxk))
    if scores is not None:
        vid_scores = np.zeros((N, scores.shape[1]))
    else:
        vid_scores = None
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
            vid_score = np.mean(clip_score, axis=0)
            vid_scores[i] = vid_score
            pred[i] = np.argsort(vid_score)[::-1][:maxk]
            # pred[i] = np.argmax(vid_score)
            gt[i] = dataloader_val.get_all_classes()[vid]  # dataloader_val.classes[vid] 
        if features is not None:
            vid_features[i] = np.mean(features[clip_ix], axis=0)

    return gt, pred, len_clip, vid_features, vid_scores


def class_balanced_acc(gt, pred, num_classes, topk=[1]):
    # assumes gt/pred are from [0, n)
    # Removing the following since sometimes not all classes exist in test set
    # unique_labels = np.unique(gt)
    # n = len(unique_labels)
    n = num_classes
    num_correct = np.zeros((n, len(topk)))
    num_labels = np.zeros(n)
    for i in range(len(gt)):
        for ki, k in enumerate(topk):
            if gt[i] in pred[i, :k]:
                num_correct[int(gt[i]), ki] += 1
        num_labels[int(gt[i])] += 1
    existing_classes = np.where(num_labels != 0)[0]
    print(f"{len(existing_classes)}/{num_classes} classes have test samples")
    acc = []
    for ki, k in enumerate(topk):
        val = (num_correct[existing_classes, ki] / num_labels[existing_classes]).mean()
        acc.append(100 * val)
    save_test_classes = False
    if save_test_classes:
        import pickle as pkl

        word_data_pkl = "bsldict/subtitles/data/words_mouthing0.8_1064_20.02.21.pkl"
        word_data = pkl.load(open(word_data_pkl, "rb"))
        nonzero_test_classes = "bsldict/subtitles/data/test_words_m0.9.txt"
        with open(nonzero_test_classes, "w") as f:
            for c in existing_classes:
                line = f'{c} {word_data["words"][c]}\n'
                f.write(line)
    return acc


def get_acc(gt, pred, topk=[1]):
    assert 1 in topk
    num_all = len(gt)
    accuracy = []
    for k in topk:
        is_correct = [gt[i] in pred[i, :k] for i in range(num_all)]
        num_correct = sum(is_correct)
        # num_correct = (gt == pred[:, k]).sum()
        acc = 100 * num_correct / num_all
        accuracy.append(acc)
        if k == 1:
            num_correct1 = num_correct
            confmat = sklearn.metrics.confusion_matrix(gt, pred[:, 0])
    return accuracy, confmat, num_correct1, num_all


def viz_confmat(confmat, accuracy, categories, save_path=None):
    # shorten category names
    for i, c in enumerate(categories):
        if len(c) > 20:
            categories[i] = c[:20]
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(confmat)
    # plt.colorbar()
    plt.xticks(np.arange(len(categories)), categories, rotation="vertical", fontsize=8)
    plt.yticks(np.arange(len(categories)), categories, fontsize=8)
    plt.xlabel("predicted labels")
    plt.ylabel("true labels")
    plt.title("Accuracy = f{accuracy[0]:.1f}")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def get_dataloader(args):
    common_kwargs = {
        "stride": args.stride,
        "inp_res": args.inp_res,
        "resize_res": args.resize_res,
        "num_in_frames": args.num_in_frames,
        "gpu_collation": args.gpu_collation,
    }
    if args.datasetname == "bsl1k":
        loader = datasets.BSL1K(
            setname=args.test_set,
            evaluate_video=args.evaluate_video,
            word_data_pkl=args.word_data_pkl,
            mouthing_window_secs=args.mouthing_window_secs,
            input_type=args.input_type,
            pose_keys=args.pose_keys,
            bsl1k_pose_subset=args.bsl1k_pose_subset,
            mask_rgb=args.mask_rgb,
            mask_type=args.mask_type,
            mask_prob=args.mask_prob,
            bsl1k_anno_key=args.bsl1k_anno_key,
            num_last_frames=args.bsl1k_num_last_frames,
            **common_kwargs,
        )
    elif args.datasetname == "wlasl":
        loader = datasets.WLASL(
            root_path="data/wlasl",
            setname=args.test_set,
            evaluate_video=args.evaluate_video,
            ram_data=args.ram_data,
            **common_kwargs,
        )
    elif args.datasetname == "msasl":
        loader = datasets.MSASL(
            root_path="data/msasl",
            setname=args.test_set,
            evaluate_video=args.evaluate_video,
            ram_data=args.ram_data,
            **common_kwargs,
        )
    else:
        raise ValueError(f"{args.datasetname} not recognised in evaluate.py")
    return loader


def evaluate(args, dataloader_val, plog):
    with_scores = True
    with_features = args.save_features
    exp_root = args.checkpoint
    scores = None
    if with_scores:
        scores_file = f"{exp_root}/preds.mat"
        plog.info(f"Loading from {scores_file}")
        scores_dict = sio.loadmat(scores_file)
        scores = scores_dict["preds"]
        plog.info(scores.shape)  # e.g. [32558, 60]
        assert scores.shape[0] == len(dataloader_val.valid)
    features = None
    if with_features:
        plog.info("Loading features.mat")
        features_file = f"{exp_root}/features.mat"
        features_dict = sio.loadmat(features_file)
        features = features_dict["preds"]

    # Aggregate the clips of each video, compute the GT/Pred/Feature for each video.
    gt, pred, len_clip, vid_features, vid_scores = aggregate_clips(
        dataloader_val, topk=args.topk, scores=scores, features=features
    )

    if vid_features is not None:
        plog.info("Saving vid_features.mat.")
        clip_gt = features_dict["clip_gt"]  # np.asarray(dataloader_val.classes)[dataloader_val.valid]
        clip_ix = features_dict["clip_ix"]  # dataloader_val.valid
        video_names = features_dict["video_names"]  # dataloader_val.videos
        sio.savemat(
            f"{exp_root}/vid_features.mat",
            {
                "vid_features": vid_features,
                "clip_gt": clip_gt,
                "clip_ix": clip_ix,
                "video_names": video_names,
                "video_gt": gt,
            },
        )

    if with_scores:
        accuracy, confmat, num_correct, num_all = get_acc(gt, pred, topk=args.topk)
        cb_acc = class_balanced_acc(gt, pred, args.num_classes, topk=args.topk)

        # Save to be able to reproduce
        results_file = f"{exp_root}/results.mat"
        plog.info(f"Saving to {results_file}")
        results_dict = {
            "accuracy": accuracy,
            "confmat": confmat,
            "num_correct": num_correct,
            "num_all": num_all,
            "gt": gt,
            "pred": pred,
            "vid_scores": vid_scores,
            "test_index": dataloader_val.valid,
            "test_t_beg": dataloader_val.t_beg,
            "videos": scores_dict["video_names"],  # dataloader_val.videos,
            "cb_acc": cb_acc,
        }
        sio.savemat(results_file, results_dict)

        # Print to file for better readability
        with open(results_file.replace(".mat", ".txt"), "w") as f:
            f.write(f"Histogram of clip lengths ({scores.shape[0]} clips):\n")
            f.write(
                np.array2string(np.histogram(len_clip, bins=np.unique(len_clip))[0])
                + "\n\n"
            )
            f.write("Histogram of GT - Pred per category:\n")
            for i, classix in enumerate(np.unique(gt)):
                f.write(f"{i} - {int(classix)}: G({(gt == classix).sum()}) - P({(pred[:, 0] == classix).sum()})\n")
            f.write("Confusion matrix:\n")
            # f.write(np.array2string(confmat, threshold=np.inf, max_line_width=np.inf) + '\n\n')
            f.write(np.array2string(confmat) + "\n\n")
            for ki, k in enumerate(args.topk):
                f.write(
                    f"Accuracy-k{k}: {accuracy[ki]:.2f}% ({num_correct}/{num_all})\n"
                )
                f.write(
                    f"Class-balanced Accuracy-k{k}: {cb_acc[ki]:.2f}%\n"
                )

        # Std out
        plog.info(f"Histogram of clip lengths ({scores.shape[0]} clips):")
        plog.info(np.histogram(len_clip, bins=np.unique(len_clip))[0])
        plog.info("Histogram of GT - Pred per category:\n")
        for i, classix in enumerate(np.unique(gt)):
            plog.info(f"{i} - {int(classix)}: G({(gt == classix).sum()}) - P({(pred[:, 0] == classix).sum()})")
        plog.info("Confusion matrix:")
        plog.info(confmat)
        for ki, k in enumerate(args.topk):
            plog.info(
                f"Accuracy-k{k}: {accuracy[ki]:.2f}% ({num_correct}/{num_all})"
            )
            plog.info(
                f"Class-balanced Accuracy-k{k}: {cb_acc[ki]:.2f}%")
        viz_confmat(
            confmat,
            accuracy,
            dataloader_val.class_names,
            save_path=f"{exp_root}/confmat.png",
        )
        return scores, accuracy


if __name__ == "__main__":
    args = opts.parse_opts()
    args.evaluate_video = 1
    args.test_set = "test"
    plog = logging.getLogger("eval")
    dataloader_val = get_dataloader(args)
    evaluate(args, dataloader_val, plog)
