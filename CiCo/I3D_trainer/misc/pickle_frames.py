"""A small utility to store video content as pickle files, rather than compressed
mp4s to allow faster debugging.

Sample usage:
%run -i misc/pickle_frames.py --refresh --limit 10 --dataset wlasl

Sample usage via yaspi:

DATASET=bsl_signdict
LIMIT=0
PROCESSES=20
CPUS_PER_TASK=$PROCESSES
NUM_PARTITIONS=20
MEMORY=64G
JOB_ARRAY_SIZE=$NUM_PARTITIONS
PARTITION=compute
PREP="pushd ${HOME}/coding/libs/pt/bsltrain"
CMD="python -u misc/pickle_frames.py --dataset=${DATASET} \
         --num_partitions ${NUM_PARTITIONS} --limit ${LIMIT} --refresh --processes 4"
python yaspi.py --job_name=pickle-frames \
                --cmd="${CMD}" \
                --job_array_size=${JOB_ARRAY_SIZE} \
                --cpus_per_task=${CPUS_PER_TASK} \
                --mem=${MEMORY} \
                --recipe=cpu-proc \
                --prep="${PREP}" \
                --partition=${PARTITION} \
                --refresh_logs \
                --throttle_array 10

Aggregation
%run -i misc/pickle_frames.py --dataset=bsl_signdict \
         --num_partitions 20 --refresh --aggregate
"""
import io
import os
import time
import pickle
import socket
import argparse
import multiprocessing as mp
from pathlib import Path

import tqdm
import numpy as np
import humanize
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from zsvision.zs_iterm import zs_dispFig
from zsvision.zs_utils import BlockTimer
from zsvision.zs_multiproc import starmap_with_kwargs

import cv2
from utils.imutils import resize_generic


def parse_video_content(
    video_idx, video_path, store_compressed, vis, resize_res, total_videos, processes
):
    frames = []
    markers = 100
    if processes > 1 and video_idx % int(max(total_videos, markers) / markers) == 0:
        pct = 100 * video_idx / total_videos
        print(f"processing {video_idx}/{total_videos} [{pct:.1f}%] [{video_path}]")
    cap = cv2.VideoCapture(str(video_path))
    orig_dims = None
    while True:
        ret, rgb = cap.read()
        if ret:
            # BGR (OpenCV) to RGB
            rgb = rgb[:, :, [2, 1, 0]]
            if store_compressed:
                buffer = io.BytesIO()
                im = Image.fromarray(rgb)
                orig_dims = im.size
                resized = im.resize((resize_res, resize_res))
                resized.save(buffer, format="JPEG", quality=store_compressed)
                if vis:
                    plt.imshow(resized)
                    zs_dispFig()
                rgb = buffer
            else:
                # apply Gul-style preproc
                iH, iW, iC = rgb.shape
                if iW > iH:
                    nH, nW = resize_res, int(resize_res * iW / iH)
                else:
                    nH, nW = int(resize_res * iH / iW), resize_res
                orig_dims = (iH, iW)
                rgb = resize_generic(rgb, nH, nW, interp="bilinear")
            frames.append(rgb)
        else:
            break
    cap.release()
    if not store_compressed:
        frames = np.array(frames)
    store = {"data": frames, "orig_dims": orig_dims}
    if frames:
        assert orig_dims is not None, "expected resize_ratio to be set"
    return store


def store_as_pkl(
    video_dir,
    dest_path,
    vis,
    limit,
    resize_res,
    store_compressed,
    processes,
    num_partitions,
    worker_id,
):
    video_paths = list(video_dir.glob("**/*.mp4"))
    print(f"Found {len(video_paths)} videos in {video_dir}")

    if num_partitions > 1:
        video_paths = np.array_split(video_paths, num_partitions)[worker_id]

    if limit:
        video_paths = video_paths[:limit]

    data = {}
    kwarg_list = []
    for ii, video_path in enumerate(video_paths):
        kwargs = {
            "video_idx": ii,
            "vis": vis,
            "resize_res": resize_res,
            "video_path": video_path,
            "total_videos": len(video_paths),
            "store_compressed": store_compressed,
            "processes": processes,
        }
        kwarg_list.append(kwargs)

    func = parse_video_content
    if processes > 1:
        with mp.Pool(processes=processes) as pool:
            res = starmap_with_kwargs(pool=pool, func=func, kwargs_iter=kwarg_list)
        for store, kwargs in zip(res, kwarg_list):
            data[str(kwargs["video_path"])] = store
    else:
        for kwargs in tqdm.tqdm(kwarg_list):
            data[str(kwargs["video_path"])] = func(**kwargs)

    # if store_compressed:
    num_bytes = [
        sum(x.getbuffer().nbytes for x in vid["data"]) for vid in data.values()
    ]
    print(
        (
            f"[Video size] >>> avg: {humanize.naturalsize(np.mean(num_bytes))}, "
            f"max: {humanize.naturalsize(np.max(num_bytes))}, "
            f"min: {humanize.naturalsize(np.min(num_bytes))}"
        )
    )
    tic = time.time()
    print(f"Writing data to {dest_path}")
    with open(dest_path, "wb") as f:
        pickle.dump(data, f)
    duration = time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - tic))
    pickle_size = humanize.naturalsize(dest_path.stat().st_size, binary=True)
    print(f"Finished writing pickle [{pickle_size}] to disk in {duration}")


def aggregate_pkls(dest_path, dataset, partition_dir, num_partitions, limit):
    template = f"{dataset}*-of-{num_partitions:02d}.pkl"
    found = list(Path(partition_dir).glob(template))
    if limit:
        found = [x for x in found if f"limit-{limit}" in str(x)]
    print(f"Found {len(found)} pickles in {partition_dir} for {dataset}")
    assert (
        len(found) == num_partitions
    ), f"Expected {num_partitions}, found {len(found)}"
    aggregate = {}
    for path in tqdm.tqdm(found):
        with open(path, "rb") as f:
            data = pickle.load(f)
        msg = f"Expected no overlap between aggregated and partition {path}"
        assert not set(data.keys()).intersection(aggregate.keys()), msg
        aggregate.update(data)
    print(f"Writing aggregated data to {dest_path}")
    with open(dest_path, "wb") as f:
        pickle.dump(aggregate, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="bsl_signbank",
        choices=["bsl_signbank", "bsl_signdict", "bbcsl", "wlasl", "msasl"],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--resize_res", type=int, default=256)
    parser.add_argument("--task", choices=["from_vids", "resize_existing"])
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--store_compressed", type=int, default=90)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--dest_dir", default="data/pickled-videos")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_partitions", type=int, default=1)
    args = parser.parse_args()

    # set up local filesystem on the cluster
    if socket.gethostname().endswith("cluster"):
        os.system(str(Path.home() / "configure_tmp_data.sh"))
    if args.slurm:
        print(f"Running aggregation via slurm on {socket.gethostname()}")

    if args.dataset in {"bsl_signbank", "bsl_signdict", "wlasl", "msasl"}:
        tag = "videos_360h_25fps"
    elif args.dataset in {"bbcsl"}:
        tag = "videos"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    video_dir = Path("data") / args.dataset / tag
    if args.store_compressed:
        fname = args.dataset
        fname = f"{fname}-compressed-quality-{args.store_compressed}"
        fname = f"{fname}-resized-{args.resize_res}x{args.resize_res}"

    if args.limit:
        fname = f"{fname}-limit-{args.limit}"
    dest_dir = Path(args.dest_dir)

    partition_dir = dest_dir / "partitions"
    if args.num_partitions > 1 and not args.aggregate:
        dest_dir = partition_dir
        fname = f"{fname}-partition-{args.worker_id:02d}-of-{args.num_partitions:02d}"
    dest_path = dest_dir / f"{fname}.pkl"
    dest_path.parent.mkdir(exist_ok=True, parents=True)
    if dest_path.exists() and not args.refresh:
        print(f"Found existing file at {dest_path}, skipping...")
        return

    if args.aggregate:
        with BlockTimer("Aggregating monolithic pickles"):
            aggregate_pkls(
                limit=args.limit,
                dest_path=dest_path,
                dataset=args.dataset,
                partition_dir=partition_dir,
                num_partitions=args.num_partitions,
            )
    else:
        with BlockTimer("Storing frames in monolithic pickle"):
            store_as_pkl(
                vis=args.vis,
                limit=args.limit,
                video_dir=video_dir,
                dest_path=dest_path,
                processes=args.processes,
                resize_res=args.resize_res,
                worker_id=args.worker_id,
                store_compressed=args.store_compressed,
                num_partitions=args.num_partitions,
            )


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
