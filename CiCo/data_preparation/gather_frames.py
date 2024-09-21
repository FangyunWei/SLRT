"""Utility script for convering phoenix2014T frames into videos.
"""
import argparse
import os


def mkdir_p(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(input_path,output_path):
    sets = ["train", "dev", "test"]
    for s in sets:
        frames_set = os.path.join(input_path, s)
        output_dir_videos = os.path.join(output_path, "videos", s)
        mkdir_p(output_dir_videos)
        for v in os.listdir(frames_set):
            frames_dir = os.path.join(frames_set, v)
            mp4_path = os.path.join(output_dir_videos, f"{v}.mp4")
            if not os.path.exists(mp4_path):
                cmd_ffmpeg = (
                    f"ffmpeg -y -threads 8 -r 25 -i " 
                    f"{os.path.join(frames_dir, 'images%04d.png')}"
                    f" -c:v h264 -pix_fmt yuv420p -crf 23 "
                    f"{mp4_path}"
                    ""
                )
                os.system(cmd_ffmpeg)


if __name__ == "__main__":
    description = (
        "Helper script for combining the original Phoenix frames into mp4 videos."
    )
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--input_path",
        type=str,
        default="PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px",
        help="Path to Phoenix data.",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="PHOENIX-2014-T-release-v3/PHOENIX-2014-T",
        help="Path to Phoenix data.",
    )
    main(**vars(p.parse_args()))
