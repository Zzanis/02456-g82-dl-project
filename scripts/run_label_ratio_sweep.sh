#!/bin/bash
#BSUB -J dl_project_g82_label_ratio[1-3]
#BSUB -q gpuv100
#BSUB -W 1:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1"
# #BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -B
#BSUB -N
#BSUB -o logs/label_ratio_%J_%I.out
#BSUB -e logs/label_ratio_%J_%I.err

DL_PROJECT_CONFIG_FILE=${DL_PROJECT_CONFIG_FILE:?Missing variable}
source $DL_PROJECT_CONFIG_FILE

REPO=${DL_PROJECT_REPO_DIR:?Missing variable}
RUN_DIR=${DL_PROJECT_RUN_DIR:-$DL_PROJECT_REPO_DIR}

LABELED_UNLABELED_RATIO_VALUES=(0.5 0.25 0.125) # 1:2, 1:4, 1:8
LABELED_UNLABELED_RATIO=${LABELED_UNLABELED_RATIO_VALUES[$((LSB_JOBINDEX-1))]}

echo "Project repo dir: $REPO"
echo "Project run dir: $RUN_DIR"
echo "LABELED_UNLABELED_RATIO: $LABELED_UNLABELED_RATIO"

mkdir -p $RUN_DIR
cd $RUN_DIR

python=python3.13

module load python3/3.13.5
module load cuda/12.8.0

source $REPO/.venv/bin/activate

python $REPO/src/run.py dataset.init.labeled_to_unlabeled_ratio=$LABELED_UNLABELED_RATIO $GNN_RUN_OPTS
