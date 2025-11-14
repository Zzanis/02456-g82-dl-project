#!/bin/bash
# Learning rate sweep script - submits multiple jobs with different LRs
# Usage: ./scripts/submit_lr_sweep.sh

# Learning rates to test
LR_VALUES=(0.0001 0.001 0.01)

echo "Submitting learning rate sweep..."
echo "Testing LRs: ${LR_VALUES[@]}"

for lr in "${LR_VALUES[@]}"; do
    echo "Submitting job with LR=$lr"
    sbatch scripts/run_experiment.sh \
        trainer.init.optimizer.lr=$lr \
        seed=0
done

echo ""
echo "All jobs submitted!"
echo "Check status with: squeue -u \$USER"
echo "View results at: https://wandb.ai/DL-gr82/semi-supervised-ensembles"
