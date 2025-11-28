#!/bin/bash
# BSUB job array: submit with `bsub < scripts/run_lr_sweep.sh`
# Adjust learning rates in LR_VALUES below.
#BSUB -J dl_project_g82_cps_loss_weight[1-15]
#BSUB -q gpuv100
#BSUB -W 1:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -B
#BSUB -N
#BSUB -o logs/cps_loss_weight_%J_%I.out
#BSUB -e logs/cps_loss_weight_%J_%I.err

DL_PROJECT_CONFIG_FILE=${DL_PROJECT_CONFIG_FILE:?Missing variable}
source $DL_PROJECT_CONFIG_FILE

REPO=${DL_PROJECT_REPO_DIR:?Missing variable}
RUN_DIR=${DL_PROJECT_RUN_DIR:-$DL_PROJECT_REPO_DIR}

# CPS_LOSS_WEIGHT_VALUES=(0 0.5 1 1.5 2 5)
CPS_LOSS_WEIGHT_VALUES=(0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 4 5)
CPS_LOSS_WEIGHT=${CPS_LOSS_WEIGHT_VALUES[$((LSB_JOBINDEX-1))]}

echo "Project repo dir: $REPO"
echo "Project run dir: $RUN_DIR"
echo "CPS_LOSS_WEIGHT: $CPS_LOSS_WEIGHT"

mkdir -p $RUN_DIR
cd $RUN_DIR

python=python3.13

module load python3/3.13.5
module load cuda/12.8.0

source $REPO/.venv/bin/activate

python $REPO/src/run.py trainer.init.cps_loss_weight=$CPS_LOSS_WEIGHT $GNN_RUN_OPTS
