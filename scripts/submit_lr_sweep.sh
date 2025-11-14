#!/bin/bash
# Learning rate sweep script - submits job array with different LRs
# Usage: ./scripts/submit_lr_sweep.sh

# Learning rates to test (must match array in run_experiment.sh)
LR_VALUES=(0.0001 0.0005 0.001 0.005 0.01)
NUM_LRS=${#LR_VALUES[@]}

echo "Submitting learning rate sweep as job array..."
echo "Testing LRs: ${LR_VALUES[@]}"
echo "Number of jobs: $NUM_LRS"
echo ""

# Submit job array - LSF will create one job per array index
# LSB_JOBINDEX will be 1, 2, 3, 4, 5 for each job
bsub -J "lr_sweep[0-${NUM_LRS-1}]" < scripts/run_experiment.sh

echo ""
echo "Job array submitted!"
echo "Check status with: bjobs"
echo "Check individual jobs: bjobs -J 'lr_sweep[*]'"
echo "View results at: https://wandb.ai/DL-gr82/semi-supervised-ensembles"
