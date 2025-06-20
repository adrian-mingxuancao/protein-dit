#!/bin/bash
#SBATCH -p general
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH -o protein_dit_%j.out
#SBATCH -e protein_dit_%j.err

# Exit on error
set -e

# Initialize conda
eval "$(~/miniconda3/bin/conda shell.bash hook)" || { echo "Failed to initialize conda"; exit 1; }
conda activate graph_md4 || { echo "Failed to activate conda environment"; exit 1; }

# Configure GPU environment
export CUDA_VISIBLE_DEVICES=0,1

# Set working directory
cd /home/caom/AID3/protein-dit || { echo "Failed to change to working directory"; exit 1; }

# Add protein-dit to Python path
export PYTHONPATH=/home/caom/AID3/protein-dit:$PYTHONPATH

# Create necessary directories
mkdir -p logs/protein_diffusion/checkpoints
mkdir -p logs/protein_diffusion/wandb

# Check if data files exist
for file in "data/protein_train/processed/protein_train_split.pt" \
            "data/protein_train/processed/protein_val_split.pt" \
            "data/protein_train/processed/protein_test_split.pt"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required data file $file not found"
        exit 1
    fi
done

# Resume training logic
RESUME_CKPT=""
if [ -n "$1" ]; then
    # If checkpoint path provided as argument
    RESUME_CKPT="$1"
    echo "Resuming from provided checkpoint: $RESUME_CKPT"
elif [ -f "logs/protein_diffusion/checkpoints/last.ckpt" ]; then
    # If last checkpoint exists, resume from it
    RESUME_CKPT="logs/protein_diffusion/checkpoints/last.ckpt"
    echo "Resuming from last checkpoint: $RESUME_CKPT"
else
    # Find the best checkpoint by validation loss
    BEST_CKPT=$(find logs/protein_diffusion/checkpoints -name "*.ckpt" -type f 2>/dev/null | head -1)
    if [ -n "$BEST_CKPT" ]; then
        RESUME_CKPT="$BEST_CKPT"
        echo "Resuming from best checkpoint: $RESUME_CKPT"
    else
        echo "No checkpoint found, starting fresh training"
    fi
fi

# Set wandb environment variables
export WANDB_API_KEY="264f2fc6de81df8451a22c5f50c0d5adca1ae0f7"
export WANDB_ENTITY="caom"
export WANDB_PROJECT="protein_diffusion"

# Print environment info
echo "Python path: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Resume checkpoint: $RESUME_CKPT"
echo "Starting training..."

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --list-gpus || echo "nvidia-smi not available"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs requested: 2"

# Set Python to unbuffered output
export PYTHONUNBUFFERED=1

# Build command with resume logic
CMD="python -u main.py \
    --config-name=train_config \
    ++train.batch_size=32 \
    ++train.max_epochs=100 \
    ++train.num_workers=4 \
    ++train.lr=1e-4 \
    ++train.weight_decay=1e-5 \
    ++train.gradient_clip_val=1.0 \
    ++train.early_stopping_patience=10 \
    ++train.accelerator=gpu \
    ++train.devices=2 \
    ++train.precision=32 \
    ++train.limit_val_batches=2 \
    ++train.num_sanity_val_steps=1 \
    ++train.accumulate_grad_batches=1 \
    ++model.drop_condition=0.1 \
    ++model.diffusion_steps=1000 \
    ++model.transition_model.type=marginal \
    ++model.transition_model.x_classes=20 \
    ++model.transition_model.e_classes=5 \
    ++model.transition_model.p_classes=20 \
    ++model.transition_model.y_classes=3 \
    ++logging.save_dir=logs/protein_diffusion \
    ++logging.log_every_n_steps=50 \
    ++logging.val_check_interval=1.0 \
    ++logging.use_wandb=true \
    ++logging.wandb_project=protein_diffusion \
    ++logging.wandb_name=protein_dit_run_${SLURM_JOB_ID} \
    ++dataset.source=alphafold_train \
    ++dataset.edge_method=knn \
    ++dataset.k=16"

# Add resume checkpoint if available
if [ -n "$RESUME_CKPT" ]; then
    CMD="$CMD ++general.resume=$RESUME_CKPT"
fi

# Run training with proper output redirection
echo "Executing: $CMD"
eval $CMD > >(tee -a training.log) 2> >(tee -a training.log >&2) 