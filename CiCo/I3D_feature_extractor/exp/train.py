from create_exp import run_cmd


if __name__ == "__main__":
    extra_args = """
        --asl_dataset wlasl \\
        --init_cross_language asl_with_bsl \\
        --num-classes 2000 \\
        --num_in_frames 64 \\
        --pretrained misc/pretrained_models/bsl1k_mouth_masked_ppose.pth.tar \\
    """
    run_cmd(
        dataset="wlasl",
        subfolder="my_experiment",
        extra_args=extra_args,
        running_mode="train",
        num_gpus=1,
        jobsub=False,
        refresh=True,
    )
