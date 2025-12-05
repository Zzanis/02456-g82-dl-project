#!/bin/bash
#BSUB -q gpuv100
#BSUB -J run_baseline_lr
#BSUB -R "rusage[mem=3GB]"
#BSUB -B
#BSUB -N
#BSUB -n 4
# GPU & placement
#BSUB -R "select[gpu]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1"
# Time & logs
#BSUB -W 00:30
#BSUB -o /zhome/3d/c/222266/02456-g82-dl-project/hpc_logs/gcn_%J.out
#BSUB -e /zhome/3d/c/222266/02456-g82-dl-project/hpc_logs/gcn_%J.err

set -euo pipefail

module purge
module load python3/3.10.12
module load gcc
source "/zhome/3d/c/222266/ComputationalTools/AmazonReview/.venv/bin/activate"

lambda_max=(0 0.01 0.03 0.05 0.08 1 1.5 )
lambda=${lambda_max[$((LSB_JOBINDEX-1))]}

#python3 src/run.py logger.name=sage trainer=semi-supervised-ensemble model=sage 

#python3 src/run.py logger.name=sage_residual_lr0.01_wd0.005_256nodes_4layers trainer=semi-supervised-ensemble model=sage_residual trainer.init.optimizer.lr=0.01 trainer.init.optimizer.weight_decay=0.005 

#python3 run.py logger.name=basic_model_meanTeacher_allData+gradientClipping trainer=semi-supervised-meanteacher1:1_allData trainer.num_models=1

#python3 run.py logger.name=basic_model_meanTeacher1:1-lr trainer=semi-supervised_meanteacher1:1 trainer.num_models=1 trainer.init.optimizer.lr=1e-3
python3 run.py logger.name=basic_model_meanTeacher1:1-A-bestShot2 trainer=semi-supervised-ensembleA trainer.num_models=1 trainer.init.optimizer.lr=0.001 trainer.init.lambda_max=0.05 trainer.init.lambda_ramp_epochs=100 trainer.init.ema_init=0.995 trainer.init.ema_target=0.999  trainer.init.ema_ramp_epochs=100 trainer.train.total_epochs=350 hydra.job.chdir=false