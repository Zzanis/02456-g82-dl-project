#!/bin/bash
#BSUB -q gpua100
#BSUB -J run_sage
#BSUB -n 8
#BSUB -R "rusage[mem=24000]"
#BSUB -M 24000
# GPU & placement
#BSUB -R "select[gpu]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1"
# Time & logs
#BSUB -W 01:00
#BSUB -o /zhome/3d/c/222266/02456-g82-dl-project/hpc_logs/gcn_%J.out
#BSUB -e /zhome/3d/c/222266/02456-g82-dl-project/hpc_logs/gcn_%J.err

set -euo pipefail

module purge
module load python3/3.10.12
module load gcc
source "/zhome/3d/c/222266/ComputationalTools/AmazonReview/.venv/bin/activate"

#python3 src/run.py logger.name=sage trainer=semi-supervised-ensemble model=sage 

#python3 src/run.py logger.name=sage_residual_lr0.01_wd0.005_256nodes_4layers trainer=semi-supervised-ensemble model=sage_residual trainer.init.optimizer.lr=0.01 trainer.init.optimizer.weight_decay=0.005 



python3 run.py logger.name=basic_model_meanTeacher_1to1_emadecay30+0.01 trainer=semi-supervised_meanteacher1:1 trainer.num_models=1 trainer.init.consistency_rampup_epochs=30 trainer.init.consistency_weight=0.01

