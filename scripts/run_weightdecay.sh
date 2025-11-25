#!/bin/bash
# BSUB job array: submit with `bsub < scripts/run_lr_sweep.sh`
# Adjust learning rates in LR_VALUES below.
#BSUB -J weightdecay[1-4]
#BSUB -q gpuv100
#BSUB -W 00:15
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1"
#BSUB -o logs/weightdecay_%J_%I.out
#BSUB -e logs/weightdecay_%J_%I.err

module purge
module load python3/3.13.8
module load cuda/12.1
source /zhome/c5/9/156511/DL-project/.venv/bin/activate

export WANDB_MODE=online
mkdir -p logs wandb_tmp

LR_VALUES=(0.01 0.001 0.0005 0.0001)
LR=${LR_VALUES[$((LSB_JOBINDEX-1))]}

python src/run.py trainer.init.optimizer.weight_decay=$LR "$@"