#!/bin/bash
#BSUB -J miglec-dl
#BSUB -q gpuv100
#BSUB -R "rusage[mem=4GB]"
#BSUB -B
#BSUB -N
#BSUB -o /zhome/2b/a/223921/runs/02456-g82-dl-project/Output_%J.out
#BSUB -e /zhome/2b/a/223921/runs/02456-g82-dl-project/Output_%J.err
#BSUB -W 2:00 
# Need to specify at least 4 cores when using GPU queue
#BSUB -n 4
# Select the resources: 1 gpu in exclusive process mode
#BSUB -gpu "num=1"
##BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"


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
