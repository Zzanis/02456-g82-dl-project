#!/bin/bash
#BSUB -J dl_project_g82_ncps_lr[1-5]
#BSUB -q gpuv100
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1"
# #BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -B
#BSUB -N
#BSUB -o logs/dl_project_g82_ncps_lr%J_%I.out
#BSUB -e logs/dl_project_g82_ncps_lr%J_%I.err

DL_PROJECT_CONFIG_FILE=${DL_PROJECT_CONFIG_FILE:?Missing variable}
source $DL_PROJECT_CONFIG_FILE

REPO=${DL_PROJECT_REPO_DIR:?Missing variable}
RUN_DIR=${DL_PROJECT_RUN_DIR:-$DL_PROJECT_REPO_DIR}

LR_VALUES=(0.0001 0.001 0.005 0.01 0.03)
LR=${LR_VALUES[$((LSB_JOBINDEX-1))]}

echo "Project repo dir: $REPO"
echo "Project run dir: $RUN_DIR"

mkdir -p $RUN_DIR
cd $RUN_DIR

python=python3.13

module load python3/3.13.5
module load cuda/12.8.0

source $REPO/.venv/bin/activate

python $REPO/src/run.py +experiment=ncps-same-arch trainer.init.optimizer.0.lr=$LR $GNN_RUN_OPTS
