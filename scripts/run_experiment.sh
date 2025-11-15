#!/bin/bash
#SBATCH --job-name=gnn_qm9
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu  # Adjust for your HPC

# Load required modules (adjust for your HPC system)
module purge
module load python/3.10.12
module load cuda/12.1

# Activate virtual environment
source ~/my_venv/bin/activate

# Set WandB mode (use offline if compute nodes don't have internet)
# export WANDB_MODE=offline
# export WANDB_DIR=$HOME/wandb_logs

# Change to project directory
cd ~/02456-g82-dl-project/

# Print configuration info
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================"

# Run experiment with any command-line arguments
python src/run.py "$@"

echo "================================"
echo "End time: $(date)"
echo "Job finished!"
echo "================================"

# Sync WandB logs if using offline mode
# wandb sync $WANDB_DIR/wandb/offline-run-*
