CONFIGSTR="configs/polIter_rho1bSft2_vineppo_GSM8K.jsonnet"
# CONFIGSTR="configs/polIter_rho1bSft2_vineppo_GSM8K.jsonnet, configs/trainers/devBz16.jsonnet"
APP_DIRECTORY="experiments/exp"

export APP_SEED="2746318213"
export WANDB_RUN_ID="2" # Optional

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# NUM_GPUS=1
# Run the training
deepspeed --no_local_rank --num_gpus=$NUM_GPUS  \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_iteration_loop
