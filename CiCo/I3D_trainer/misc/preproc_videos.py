"""A small utility to preprocess the videos to use square frames (this resize will
break the aspect ratio for storage, but the op can be undone during training).

Sample usage:
python misc/preproc_videos.py --dataset BSLCP_raw --yaspify --num_partitions=10
"""
import socket
import argparse
import getpass
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from beartype import beartype

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from beartype.cave import NoneType
from zsvision.zs_iterm import zs_dispFig
from zsvision.zs_multiproc import starmap_with_kwargs


def video_path2id(video_path):
    """Convert bbcsl format video paths into ids - these are keys of the form:
    "<show>--<episode>".

    Args:
        video_path (Path): the location of the video

    Return:
        (str): A bbcsl dataset id key.
    """
    assert "bbcsl" in str(video_path), "video_path2id is only supported for bbcsl data"
    return f"{video_path.parent.parent.stem}--{video_path.parent.stem}"


def resize_video_content(
    vis,
    video_idx,
    dest_path,
    resize_res,
    video_path,
    total_videos,
    processes,
    progress_markers,
):
    progress_interval = int(max(total_videos, progress_markers) / progress_markers)
    dest_path.parent.mkdir(exist_ok=True, parents=True)
    if processes > 1 and video_idx % progress_interval == 0:
        pct = progress_markers * video_idx / total_videos
        print(f"processing {video_idx}/{total_videos} [{pct:.1f}%] [{video_path}]")

    cmd = f"ffmpeg -i {video_path} -y -vf scale={resize_res}:{resize_res} {dest_path}"
    os.system(cmd)

    if vis:
        cap = cv2.VideoCapture(str(dest_path))
        ret, rgb = cap.read()
        if ret:
            # BGR (OpenCV) to RGB
            rgb = rgb[:, :, [2, 1, 0]]
            if vis:
                plt.imshow(rgb)
                print(f"Showing first frame from video")
                plt.savefig("zz-first-frame.png")
                import ipdb; ipdb.set_trace()
                # zs_dispFig()
        cap.release()


@beartype
def resize_videos(
    src_video_dir: Path,
    dest_video_dir: Path,
    relevant_ids_path: (Path, NoneType),
    vis: bool,
    limit: int,
    suffix: str,
    refresh: bool,
    processes: int,
    resize_res: int,
    worker_id: int,
    progress_markers: int,
    num_partitions: int,
    exclude_pattern: (str, NoneType),
):
    video_paths = list(src_video_dir.glob(f"**/*{suffix}"))
    print(f"Found {len(video_paths)} videos in {src_video_dir}")

    if relevant_ids_path:
        with open(relevant_ids_path, "r") as f:
            relevant_ids = set(f.read().splitlines())
        video_paths = [x for x in video_paths if video_path2id(x) in relevant_ids]
        print(f"Filtered to {len(video_paths)} videos using relevant-id list")

    if exclude_pattern:
        pre_exclude = len(video_paths)
        video_paths = [x for x in video_paths if exclude_pattern not in x.name]
        print(f"Filtered from {pre_exclude} videos to {len(video_paths)} "
              f" by excluding the pattern: {exclude_pattern}")

    video_paths = np.array_split(video_paths, num_partitions)[worker_id]

    if limit:
        video_paths = video_paths[:limit]

    # Some source videos were re-encoded to fix meta data issues.  When these are used
    # we rename their targets to match the other videos
    remap = {"signhd-dense-fast-audio": "signhd"}

    kwarg_list = []
    for ii, video_path in enumerate(video_paths):
        dest_path = dest_video_dir / video_path.relative_to(src_video_dir)
        # We enforce that all videos are re-encoded as mp4, regardless of source format
        dest_path = dest_path.with_suffix(".mp4")
        if any([key in str(dest_path) for key in remap]):
            for src, target in remap.items():
                dest_path = Path(str(dest_path).replace(src, target))
        if dest_path.exists() and not refresh:
            print(f"Found existing video at {dest_path}, skipping")
            continue

        kwargs = {
            "vis": vis,
            "video_idx": ii,
            "processes": processes,
            "dest_path": dest_path,
            "resize_res": resize_res,
            "video_path": video_path,
            "total_videos": len(video_paths),
            "progress_markers": progress_markers,
        }
        kwarg_list.append(kwargs)

    func = resize_video_content
    if processes > 1:
        with mp.Pool(processes=processes) as pool:
            starmap_with_kwargs(pool=pool, func=func, kwargs_iter=kwarg_list)
    else:
        for kwargs in tqdm.tqdm(kwarg_list):
            func(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="bsl_signbank",
        choices=[
            "bsl_signbank",
            "bsl_signdict",
            "bbcsl",
            "msasl",
            "wlasl",
            "bbcsl_annotated",
            "bbcsl_raw",
            "BSLCP_raw",
        ],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--resize_res", type=int, default=256)
    parser.add_argument("--worker_id", default=0, type=int)
    parser.add_argument("--progress_markers", type=int, default=100)
    parser.add_argument("--yaspify", action="store_true")
    parser.add_argument("--use_gnodes", action="store_true")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--mouthing_window_secs",
        default=0,
        type=int,
        help="if given, preprocess videos from different windows.",
    )
    parser.add_argument("--num_partitions", default=1, type=int)
    parser.add_argument("--relevant_ids_path", help="if given, filter to these ids")
    parser.add_argument("--yaspi_defaults_path", default="misc/yaspi_cpu_defaults.json")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    fname_suffix = ""
    if args.mouthing_window_secs:
        assert args.dataset == "bbcsl", f"Mouthing windows are only supported for bbcsl"
        fname_suffix = f"-{args.mouthing_window_secs}sec-window-signhd"

    exclude_pattern = None
    if args.dataset in {"bsl_signbank", "bsl_signdict", "msasl", "wlasl"}:
        tag, suffix = "videos_360h_25fps", ".mp4"
    elif args.dataset in {"bbcsl", "bbcsl_annotated"}:
        fname_suffix += "-videos-fixed"
        tag, suffix = "annotated-videos-fixed", ".mp4"
    elif args.dataset == "bbcsl_raw":
        tag, suffix = "videos-mp4", "signhd-dense-fast-audio.mp4"
    elif args.dataset == "BSLCP_raw":
        tag, suffix, exclude_pattern = "videos", ".mov", "+"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    dest_fname = f"videos-resized-25fps-{args.resize_res}x{args.resize_res}"
    dataset_name2dir = {"BSLCP_raw": "BSLCP"}
    dataset_dir = dataset_name2dir.get(args.dataset, args.dataset)

    src_video_dir = Path("data") / dataset_dir / f"{tag}{fname_suffix}"
    dest_video_dir = src_video_dir.parent / f"{dest_fname}{fname_suffix}"

    if getpass.getuser() == "albanie" and socket.gethostname().endswith("cluster"):
        os.system(str(Path.home() / "configure_tmp_data.sh"))

    if args.yaspify:
        # Only import yaspi if requested
        from yaspi.yaspi import Yaspi

        with open(args.yaspi_defaults_path, "r") as f:
            yaspi_defaults = json.load(f)
        cmd_args = sys.argv
        cmd_args.remove("--yaspify")
        base_cmd = f"python {' '.join(cmd_args)}"
        job_name = f"preproc-videos-{args.num_partitions}-partitions"
        if args.use_gnodes:
            yaspi_defaults["partition"] = "gpu"
        job = Yaspi(
            cmd=base_cmd,
            job_queue=None,
            gpus_per_task=0,
            job_name=job_name,
            job_array_size=args.num_partitions,
            **yaspi_defaults,
        )
        job.submit(watch=True, conserve_resources=5)
    else:
        resize_videos(
            vis=args.vis,
            suffix=suffix,
            limit=args.limit,
            refresh=args.refresh,
            worker_id=args.worker_id,
            processes=args.processes,
            resize_res=args.resize_res,
            num_partitions=args.num_partitions,
            src_video_dir=src_video_dir,
            dest_video_dir=dest_video_dir,
            relevant_ids_path=args.relevant_ids_path,
            progress_markers=args.progress_markers,
            exclude_pattern=exclude_pattern,
        )


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
