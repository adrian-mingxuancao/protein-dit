#!/bin/bash

# Helper script to resume training from a specific checkpoint
# Usage: ./resume_training.sh [checkpoint_path]

if [ $# -eq 0 ]; then
    echo "Usage: $0 [checkpoint_path]"
    echo "Example: $0 logs/protein_diffusion/checkpoints/epoch=10-val_NLL=2.34.ckpt"
    echo "Or run without arguments to auto-detect the best checkpoint"
    exit 1
fi

CHECKPOINT_PATH="$1"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "Resuming training from checkpoint: $CHECKPOINT_PATH"

# Submit SLURM job with the checkpoint path
sbatch train.slurm "$CHECKPOINT_PATH" 