#!/bin/bash
#SBATCH --account=def-bboulet     # set account
#SBATCH --gpus-per-node=4         # Number of GPU(s) per node
#SBATCH --output=runlog/log_%j.out      # log 保存地址
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --mem=80gb               # memory per node
#SBATCH --time=1-10:10            # set the time for tasks     3 days 2 hours 1 minute 0 second for --time==3-02:01:00

module load StdEnv/2023  gcc/12.3 cuda/12.2 arrow/17.0 rust/1.70.0 python/3.10.13 git-lfs/3.4.0             # load the module
#cd /project/def-bboulet/imadlak/program/VinePPO                                                # set the path
source venv/bin/activate                                                                       # activate the env

export WANDB_MODE=offline         # set wandb to offline mode
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  
CONFIGSTR="configs/polIter_rho1bSft2_vineppo_GSM8K.jsonnet"
# CONFIGSTR="configs/polIter_rho1bSft2_vineppo_GSM8K.jsonnet, configs/trainers/devBz16.jsonnet"             # for 1 gpu training
APP_DIRECTORY="experiments/exp"

export APP_SEED="2746318213"
export WANDB_RUN_ID="2" # Optional

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# NUM_GPUS=1
# Run the training

deepspeed --no_local_rank --num_gpus=$NUM_GPUS  \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_iteration_loop



deactivate
