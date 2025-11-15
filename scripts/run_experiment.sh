#!/bin/bash
#BSUB -J dl_project_lr               
#BSUB -q gpuv100                    
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=5GB]"
#BSUB -gpu "num=1:mode=exclusive_process"  
#BSUB -o logs/learningrate_%J_%I.out          
#BSUB -e logs/learningrate_%J_%I.err         

# Load required modules (adjust for your HPC system)
module purge
module load python3/3.13.8
module load cuda/12.1

# Activate virtual environment
echo "Activating virtual environment..."
source /zhome/c5/9/156511/DL-project/.venv/bin/activate
echo "Virtual environment activated"
echo "Python location: $(which python3)"
echo "Python version: $(python3 --version)"

# Set WandB mode (try online mode - will fall back to offline if no internet)
export WANDB_MODE=online
export WANDB_DIR=/zhome/c5/9/156511/DL-project/02456-g82-dl-project/logs/wandb_logs

echo "WandB mode: $WANDB_MODE"

# Change to project directory
cd /zhome/c5/9/156511/DL-project/02456-g82-dl-project
echo "Working directory: $(pwd)"

# If this is a job array, map LSB_JOBINDEX to learning rate
if [ ! -z "$LSB_JOBINDEX" ]; then
    # Define learning rates array (must match submit_lr_sweep.sh)
    LR_VALUES=(0.0001 0.0005 0.001 0.005 0.01)
    # Get learning rate for this job index 
    LR=${LR_VALUES[$((LSB_JOBINDEX))]}
    echo "Job array index: $LSB_JOBINDEX, Using learning rate: $LR"
    EXTRA_ARGS="trainer.init.optimizer.lr=$LR"
else
    EXTRA_ARGS=""
fi

# Run experiment with any command-line arguments passed to this script
echo "Starting training with args: $EXTRA_ARGS $@"
python src/run.py $EXTRA_ARGS "$@"
EXIT_CODE=$?
echo "Training finished with exit code: $EXIT_CODE"


# Sync WandB logs if using offline mode
if [ "$WANDB_MODE" = "offline" ]; then
    echo "Syncing WandB logs..."
    wandb sync $WANDB_DIR/wandb/offline-run-* || echo "WandB sync failed or no runs to sync"
fi