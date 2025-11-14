#!/bin/bash
#BSUB -q gpuv100
#BSUB -J run_sage
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -M 32000
# GPU & placement
#BSUB -R "select[gpu]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1"
# Time & logs
#BSUB -W 08:00
#BSUB -o /zhome/3d/c/222266/gnn_intro/hpc_logs/gcn_%J.out
#BSUB -e /zhome/3d/c/222266/gnn_intro/hpc_logs/gcn_%J.err

set -euo pipefail

module purge
module load python3/3.10.12
module load gcc
source "/zhome/3d/c/222266/ComputationalTools/AmazonReview/.venv/bin/activate"

#python3 src/run.py logger.name=sage trainer=semi-supervised-ensemble model=sage 

python3 src/run.py logger.name=gcn_residual_lr0.005_wd0.002 trainer=semi-supervised-ensemble model=gcn_residual trainer.init.optimizer.lr=0.005 trainer.init.optimizer.weight_decay=0.002