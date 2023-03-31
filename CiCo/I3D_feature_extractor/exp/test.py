from create_exp import run_cmd


if __name__ == "__main__":
    extra_args = """
        --num-classes 2000 \\
        --num_in_frames 64 \\
        --save_features 1 \\
        --include embds 1 \\
        --test_set test \\
    """
    run_cmd(
        dataset="wlasl",
        subfolder="my_experiment",
        extra_args=extra_args,
        running_mode="test",
        modelno="_050",
        test_suffix="",
        num_gpus=1,
        jobsub=False,
        refresh=False,
    )
