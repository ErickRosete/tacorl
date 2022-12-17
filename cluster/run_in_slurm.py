import argparse
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

from git import Repo
from setuptools import sandbox


def create_git_copy_in_log_dir(log_dir: Path, sandbox_install: bool = False):
    repo_src_dir = Path(__file__).absolute().parents[1]
    repo_target_dir = log_dir / "tacorl_repo"
    create_git_copy(repo_src_dir, repo_target_dir, sandbox_install=sandbox_install)
    return repo_target_dir


def create_git_copy(repo_src_dir, repo_target_dir, sandbox_install: bool = False):
    repo = Repo(repo_src_dir)
    repo.clone(repo_target_dir)
    if sandbox_install:
        orig_cwd = os.getcwd()
        os.chdir(repo_target_dir)
        sandbox.run_setup("setup.py", ["develop", "--install-dir", "."])
        os.chdir(orig_cwd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python-file",
        type=str,
        default="scripts/train.py",
        help="Python script file to run",
    )
    parser.add_argument(
        "--bash-template",
        type=str,
        default="slurm_template.sh",
        help="Bash template to overwrite",
    )

    parser.add_argument("--conda-env", type=str, default="tacorl")
    parser.add_argument(
        "--exp-name",
        type=str,
        default="train",
        help="The job name and path to logging file (exp_name.log).",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs to use in each node"
    )
    parser.add_argument(
        "--partition", type=str, default="alldlc_gpu-rtx2080", help="Cluster partition"
    )
    parser.add_argument(
        "-D",
        "--working-dir",
        type=str,
        default="/work/dlclarge2/roseteb-tacorl/",
        help="Working directory",
    )
    parser.add_argument(
        "-n", "--num-nodes", type=int, default=1, help="Number of nodes to use."
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="Aborts cloning the repository to the log dir",
    )
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="The job allocation can not share nodes with other running jobs ",
    )
    parser.add_argument(
        "--sandbox-install",
        action="store_true",
        help="Install sandbox after creating a git repo copy",
    )
    args, unknownargs = parser.parse_known_args()
    hydra_args = " ".join(unknownargs)
    return args, hydra_args


def save_slurm_bash_file(
    bash_file_content: str = "",
    filename: str = "",
    save_dir: Path = Path(__file__).parents[0],
):
    bash_filename = str((save_dir / filename).resolve())
    with open(bash_filename, "w") as f:
        f.write(bash_file_content)
    st = os.stat(bash_filename)
    os.chmod(bash_filename, st.st_mode | stat.S_IEXEC)
    return bash_filename


def submit_job(bash_filename: str = ""):
    print("Starting to submit job!")
    subprocess.Popen(["sbatch", bash_filename])
    print(f"Job submitted! Script file is at: {bash_filename}")
    sys.exit(0)


def overwrite_template(
    args, template_file: Path, job_name: str = "", bash_command: str = ""
):

    JOB_OPTS = "${JOB_OPTS}"
    COMMAND_PLACEHOLDER = "${COMMAND_PLACEHOLDER}"
    LOAD_ENV = "${LOAD_ENV}"
    job_opts = {
        # "exclude": "dlcgpu19",
        "ntasks-per-node": 1,
        "partition": args.partition,
        "chdir": args.working_dir,
        "job-name": job_name,
        "output": f"logs/{job_name}/%x.%N.%j.out",
        "error": f"logs/{job_name}/%x.%N.%j.err",
        "nodes": str(args.num_nodes),
        "gres": f"gpu:{str(args.num_gpus)}",
    }
    if args.exclusive:
        job_opts["exclusive"] = ""
    job_opts_str = ""
    for key, value in job_opts.items():
        job_opts_str += f"#SBATCH --{key} {value}\n"

    load_env = "source ~/.bashrc\n"
    load_env += f"conda activate {args.conda_env}"

    with open(template_file, "r") as f:
        bash_file_content = f.read()
    bash_file_content = bash_file_content.replace(JOB_OPTS, job_opts_str)
    bash_file_content = bash_file_content.replace(LOAD_ENV, load_env)
    bash_file_content = bash_file_content.replace(COMMAND_PLACEHOLDER, bash_command)
    return bash_file_content


def get_log_dir_and_job_name(args):
    job_name = f"{args.exp_name}_{time.strftime('%m%d-%H%M', time.localtime())}"
    log_dir = Path(f"{args.working_dir}/logs/{job_name}")
    log_dir.mkdir(parents=True)
    return log_dir, job_name


def get_bash_command(
    hydra_args, log_dir: Path, repo_dir: Path, python_file="S3VP/train.py"
):
    bash_command = f"cd {str(repo_dir)}\n"
    bash_command += f"python {python_file} {hydra_args} hydra.run.dir={log_dir}"
    return bash_command


def main():
    args, hydra_args = parse_args()
    log_dir, job_name = get_log_dir_and_job_name(args)

    if args.no_clone:
        repo_dir = Path(__file__).resolve().parents[1]
    else:
        repo_dir = create_git_copy_in_log_dir(
            log_dir, sandbox_install=args.sandbox_install
        )

    bash_command = get_bash_command(
        hydra_args=hydra_args,
        log_dir=log_dir,
        repo_dir=repo_dir,
        python_file=args.python_file,
    )
    template_file = Path(__file__).resolve().parents[0] / args.bash_template
    bash_file_content = overwrite_template(
        args=args,
        template_file=template_file,
        job_name=job_name,
        bash_command=bash_command,
    )
    bash_filename = save_slurm_bash_file(
        bash_file_content=bash_file_content,
        filename="resume_script.sh",
        save_dir=log_dir,
    )
    submit_job(bash_filename)


if __name__ == "__main__":
    main()
