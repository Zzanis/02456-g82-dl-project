#!/bin/bash
# BSUB job array: submit with `bsub < scripts/run_lr_sweep.sh`
# Adjust learning rates in LR_VALUES below.
#BSUB -J lambda[1-7]
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10000]"
#BSUB -gpu "num=1"
# Time & logs
#BSUB -W 04:00
#BSUB -o /zhome/f9/1/147385/02456-g82-dl-project/hpc_logs/gcn_%J.out
#BSUB -e /zhome/f9/1/147385/02456-g82-dl-project/hpc_logs/gcn_%J.err

set -euo pipefail

module purge
module load python3/3.10.12
module load gcc
# module load cuda/12.1  # uncomment if needed
source "/zhome/f9/1/147385/my_venv/bin/activate"

# BATCH_VALUES=(50 150 200)
# BATCH=${BATCH_VALUES[$((LSB_JOBINDEX-1))]}

# python3 src/run.py \
#     trainer=mean_teacher_advanced_gcn_trainer \
#     model=advanced_gcn_graph_norm \
#     dataset.init.batch_size_train=$BATCH \
#     logger.name="A_gcn_MT_GraphNorm_batch${BATCH}" \
#     "$@"


# LR_VALUES=(0.0001 0.0005 0.001 0.005 0.01)
# LR=${LR_VALUES[$((LSB_JOBINDEX-1))]}

# python3 src/run.py \
#     trainer=mean_teacher_advanced_gcn_trainer \
#     model=advanced_gcn_graph_norm \
#     trainer.init.optimizer.lr=$LR \
#     logger.name="A_gcn_MT_GraphNorm_lr${LR}" \
#     "$@"

#  # Define the list of lambda_unsup values to sweep
# # # LAMBDA_VALUES=(0.001 0.005 0.01 0.05 0.1)
LAMBDA_VALUES=(0 0.01 0.03 0.05 0.08 1 1.5)
LAMBDA=${LAMBDA_VALUES[$((LSB_JOBINDEX-1))]}

# for LAMBDA in "${LAMBDA_VALUES[@]}"; do
#     echo "Running experiment with lambda_unsup = $LAMBDA"
python3 src/run.py \
    trainer=mean_teacher_advanced_gcn_trainer \
    model=advanced_gcn_graph_norm \
    trainer.init.lambda_max=$LAMBDA \
    logger.name="A_gcn_lambda_${LAMBDA}" \
    "$@"


