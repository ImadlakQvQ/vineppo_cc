#!/bin/bash
#SBATCH --account=def-bboulet     # set account
#SBATCH --gpus-per-node=4         # Number of GPU(s) per node
#SBATCH --output=runlog/exp.out      # log 保存地址
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --mem=80gb               # memory per node
#BATCH --time=1-02:00:00            # set the time for tasks     3 days 2 hours 1 minute 0 second for --time==3-02:01:00

module load StdEnv/2023  gcc/12.3 cuda/12.2 arrow/17.0 rust/1.70.0 python/3.10.13              # load the module
#cd /project/def-bboulet/imadlak/program/VinePPO                                                # set the path
source venv/bin/activate                                                                       # activate the env

export WANDB_MODE=offline         # set wandb to offline mode
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK     
./test.sh                 # run the tasks


deactivate                        # deactivate the env
