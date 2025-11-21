# Job Scripts

This folder contains scripts for running experiments locally and on HPC.

## Scripts

### `run_local.sh` - Local Testing
Quick testing script for your Mac with reduced settings.

**Usage:**
```bash
./scripts/run_local.sh
```

This runs 10 epochs on a small subset (1000 samples) for quick testing.

### `run_experiment.sh` - Main HPC Job Script
Template for submitting jobs to the HPC cluster.

**Before using:**
1. Update the paths in the script (search for `/path/to/`)
2. Adjust SLURM settings for your HPC (partition, modules, etc.)
3. Decide if you need offline mode for WandB

**Usage:**
```bash
# Basic run
sbatch scripts/run_experiment.sh

# With custom parameters
sbatch scripts/run_experiment.sh optimizer.init.lr=0.001

# Multiple parameters
sbatch scripts/run_experiment.sh \
    optimizer.init.lr=0.001 \
    model.init.0.hidden_channels=128 \
    seed=42
```

### `submit_lr_sweep.sh` - Learning Rate Sweep
Submits multiple jobs with different learning rates.

**Usage:**
```bash
./scripts/submit_lr_sweep.sh
```

Tests learning rates: 0.0001, 0.001, 0.01

## Customization

You can create more sweep scripts for different hyperparameters:

**Example: Hidden Channels Sweep**
```bash
#!/bin/bash
HIDDEN_VALUES=(32 64 128 256)

for hidden in "${HIDDEN_VALUES[@]}"; do
    sbatch scripts/run_experiment.sh \
        model.init.0.hidden_channels=$hidden
done
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View output
tail -f logs/output_<job_id>.log

# Cancel job
scancel <job_id>
```

## WandB Integration

Results are automatically logged to: https://wandb.ai/DL-gr82/semi-supervised-ensembles

If using offline mode on HPC, sync after jobs complete:
```bash
wandb sync ~/wandb_logs/wandb/offline-run-*
```
