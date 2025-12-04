#!/bin/bash
# BSUB job array: submit with `bsub < scripts/run_lr_sweep.sh`
# Adjust learning rates in LR_VALUES below.
#BSUB -J CPS[1-4]
#BSUB -q gpuv100
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1"
#BSUB -o logs/CPS_%J_%I.out
#BSUB -e logs/CPS_%J_%I.err

module purge
module load python3/3.13.8
module load cuda/12.1
source /zhome/c5/9/156511/DL-project/.venv/bin/activate

export WANDB_MODE=online
mkdir -p logs wandb_tmp

LR_VALUES=(0.2 0.15 0.1 0.05)
LR=${LR_VALUES[$((LSB_JOBINDEX-1))]}

python src/run.py model.init.dropout=$LR "$@"