##	Running Taco-RL scripts on a Slurm Cluster

### Running a script
We provide a utility to run scripts on a Slurm cluster.
To execute a script you just need to specify to appropriate address of the 
python file script in the `--python-file` argument.

```bash
$ cd tacorl/cluster
$ python run_in_slurm.py --python-file="tacorl/train.py"
```

You can use the following optional command line arguments for slurm:
- `--conda-env`: Specifies a conda environment.
- `--exp-name`: The job name and path to logging file (exp_name.log).
- `--num-gpus`: Number of GPUs to use in each node
- `--working-dir`: Slurm working directory
- `--no-clone`: When provided it won't copy the git repo in the log dir

Additionally you can send all hydra arguments as in the normal execution.

The script will create a new folder in the specified log dir with a date tag and the job name.
This is done *before* the job is submitted to the slurm queue.

In order to ensure reproducibility, the current state of the tacorl repository
is copied to the log directory at *submit time* and is
locally installed, such that you can schedule multiple scripts and there is no 
interference with future changes to the repository.

### Resuming a script
Every job submission creates a `resume_script.sh` script in the log folder.
To resume a training,
call `$ sh <PATH_TO_LOG_DIR>/resume_script.sh`.