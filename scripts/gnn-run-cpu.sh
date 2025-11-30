#!/bin/bash
#BSUB -J miglec-dl
#BSUB -q hpc
#BSUB -W 10:00
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -B
#BSUB -N
#BSUB -o logs/cps_loss_weight_%J_%I.out
#BSUB -e logs/cps_loss_weight_%J_%I.err

REPO=/zhome/2b/a/223921/projects/02456-g82-dl-project
RUN_DIR=/zhome/2b/a/223921/runs/02456-g82-dl-project
python=python3.13

module load python3/3.13.5
module load cuda/12.8.0

source $REPO/.venv/bin/activate
source /zhome/2b/a/223921/.dl-project-config

mkdir -p $RUN_DIR
cd $RUN_DIR

#$python -m pip install -r $REPO/requirements.txt

$python $REPO/src/run.py $GNN_RUN_OPTS
