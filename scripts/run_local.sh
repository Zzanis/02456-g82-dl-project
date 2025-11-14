#!/bin/bash
# Local testing script - runs a quick experiment with reduced settings
# Usage: ./scripts/run_local.sh

set -e  # Exit on error

echo "Running local test experiment..."

# Activate virtual environment (adjust path if needed)
source ../.venv/bin/activate

# Run with reduced settings for quick testing
python src/run.py \
    trainer.train.total_epochs=10 \
    trainer.train.validation_interval=5 \
    dataset.init.batch_size_train=32 \
    dataset.init.subset_size=1000 \
    logger.disable=true \
    "$@"

echo "Local test complete!"
