import json
import os


def set_user_specific_vars(config="config.json"):
    with open(config, "r") as f:
        config_data = json.load(f)
        code_dir = config_data["code_dir"]
        activate_env = config_data["activate_env"]
    return code_dir, activate_env


def create_jobsub_str(
    jobname, expdir, num_gpus=1, mem=96, num_workers=8, duration=96
):
    """Prepends the slurm job submission string to the bash script string. 
    """
    jobsub_str = f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH -o {expdir}/qsub_out.log
#SBATCH -e {expdir}/qsub_err.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={num_gpus * num_workers}
#SBATCH --mem={num_gpus * mem}gb
#SBATCH --time={duration}:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{num_gpus}
# -------------------------------

export PATH=$PATH

"""
    return jobsub_str


def create_cmd(
    dataset,
    subfolder,
    extra_args,
    code_dir,
    activate_env,
    running_mode="train",
    modelno="",
    test_suffix="",
    num_gpus=1,
):
    """Creates a string which will be the content of the bash script. 
    """
    expdir = os.path.join(code_dir, "checkpoint", dataset, subfolder)

    if running_mode == "test":
        expdirroot = expdir
        expdir = os.path.join(expdir, f"test{modelno}{test_suffix}")

    os.makedirs(expdir, exist_ok=True)

    cmd_str = f"""
{activate_env}
cd {code_dir}
python main.py \\
    --checkpoint {expdir} \\
    --datasetname {dataset} \\
    --num_gpus {num_gpus} \\
    -j {num_gpus * 8} \\"""

    if running_mode == "test":
        cmd_str += f"""
    -e --evaluate_video 1 \\
    --pretrained {expdirroot}/checkpoint{modelno}.pth.tar \\"""

    cmd_str += f"""{extra_args}"""
    return cmd_str, expdir


def write_script_file(expdir, cmd_str, refresh=False):
    """
        Creates the bash script to run.
        If already exists:
            either overwrites if refresh is True,
            or exits to avoid overwriting.
    """
    scriptfile = f"{expdir}/run.sh"
    if os.path.exists(scriptfile):
        if refresh:
            print(f"{scriptfile} exists, refreshing...")
        else:
            print(f"{scriptfile} exists. You can:")
            print(f"   rm {scriptfile}")
            exit()
    print(f"Creating script file {scriptfile}")
    with open(scriptfile, "w") as f:
        f.write(cmd_str)
    return scriptfile


def run_cmd(
    dataset,
    subfolder,
    extra_args,
    running_mode="train",
    modelno="",
    test_suffix="",
    num_gpus=1,
    jobsub=False,
    refresh=False,
):
    """
        Runs the job.
        1) Creates a bash script with `python main.py <args>` in it.
        2) Creates the experiment folder, and saves the bash script in it.
        2) Runs the bash script either interactively or as a job on slurm cluster.

        Args:
            dataset : Name of the dataset (bsl1k | wlasl | msasl | phoenix2014 | bslcp)
            subfolder: Name of the experiment
            extra_args: A list of additional experiment parameters
            running_mode: Determines either train-val epochs or sliding window test (train | test)
            modelno: [test mode] Suffix to `checkpoint<>.pth.tar`, last epoch by default, or the epoch number, e.g., `_050`
            test_suffix: [test mode] Suffix to prepend to the testing folder, empty by default
            num_gpus: Number of gpus
            jobsub: If True, submits the bash script as a job to the slurm cluster
            refresh: If True, overwrites any existing bash script and relaunches
        Note:
            This is optional. The training/testing code can be ran by directly
            typing `python main.py <args>` on terminal.
    """
    code_dir, activate_env = set_user_specific_vars()
    cmd_str, expdir = create_cmd(
        dataset=dataset,
        subfolder=subfolder,
        extra_args=extra_args,
        code_dir=code_dir,
        activate_env=activate_env,
        running_mode=running_mode,
        modelno=modelno,
        test_suffix=test_suffix,
        num_gpus=num_gpus,
    )

    if jobsub:
        jobname = f"{running_mode}_{dataset}_{subfolder}"
        cmd_str = create_jobsub_str(jobname, expdir, num_gpus) + cmd_str

    scriptfile = write_script_file(expdir=expdir, cmd_str=cmd_str, refresh=refresh)
    print("Running:")
    print(cmd_str)
    if jobsub:
        print("SUBMITTING JOB...")
        os.system(f"sbatch {scriptfile}")
    else:
        print("RUNNING INTERACTIVELY...")
        os.system(f"bash {scriptfile}")
