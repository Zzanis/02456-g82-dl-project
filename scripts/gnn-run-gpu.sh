#!/bin/bash
#BSUB -J miglec-DLProjectGNN
#BSUB -q gpuv100
#BSUB -R "rusage[mem=5GB]"
#BSUB -B
#BSUB -N
#BSUB -o /zhome/2b/a/223921/runs/02456-g82-dl-project/Output_%J.out
#BSUB -e /zhome/2b/a/223921/runs/02456-g82-dl-project/Output_%J.err
#BSUB -W 2:00 
# Need to specify at least 4 cores when using GPU queue
#BSUB -n 4
# Select the resources: 1 gpu in exclusive process mode
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"

# use this for unbuffered output, so that you can check in real-time
# (with tail -f Output_.out Output_.err)
# what your program was printing "on the screen"
# python3 -u helloworld.py

# use this for just piping everything into a file, 
# the program knows then, that it's outputting to a file
# and not to a screen, and also combine stdout&stderr
# python3 helloworld.py > joboutput_$LSB_JOBID.out 2>&1


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
