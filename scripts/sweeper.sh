#!/bin/bash
# BSUB job array: submit with `bsub < scripts/run_lr_sweep.sh`
# Adjust learning rates in LR_VALUES below.
#BSUB -J lr[1-3]
#BSUB -q gpuv100
#BSUB -W 00:10
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -gpu "num=1"
# Time & logs
#BSUB -W 08:00
#BSUB -o /zhome/f9/1/147385/02456-g82-dl-project/hpc_logs/gcn_%J.out
#BSUB -e /zhome/f9/1/147385/02456-g82-dl-project/hpc_logs/gcn_%J.err

set -euo pipefail

module purge
module load python3/3.10.12
module load gcc
# module load cuda/12.1  # uncomment if needed
source "/zhome/f9/1/147385/my_venv/bin/activate"

LR_VALUES=(0.0001 0.001 0.01)
LR=${LR_VALUES[$((LSB_JOBINDEX-1))]}

python3 src/run.py \
    trainer=advanced_gcn_trainer \
    model=advanced_gcn \
    trainer.init.optimizer.lr=$LR \
    logger.name="A_gcn_MT_GraphNorm_lr${LR}" \
    "$@"

#  # Define the list of lambda_unsup values to sweep
# # # LAMBDA_VALUES=(0.001 0.005 0.01 0.05 0.1)
# LAMBDA_VALUES=(0.0001 0.001 0.005 0.01 0.05 0.1)
# LAMBDA=${LAMBDA_VALUES[$((LSB_JOBINDEX-1))]}

# # for LAMBDA in "${LAMBDA_VALUES[@]}"; do
# #     echo "Running experiment with lambda_unsup = $LAMBDA"
# python3 src/run.py \
#     trainer=semi-supervised-ensemble-custom \
#     model=schnet_model \
#     trainer.init.lambda_max=$LAMBDA \
#     logger.name="schnet_lambda_${LAMBDA}" \
#     "$@"


