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
cd /home/caom/AID3/Graph-DiT || { echo "Failed to change to working directory"; exit 1; }

# Add Graph-DiT to Python path
export PYTHONPATH=/home/caom/AID3/Graph-DiT:$PYTHONPATH

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

# Set wandb environment variables
export WANDB_API_KEY="264f2fc6de81df8451a22c5f50c0d5adca1ae0f7"
export WANDB_ENTITY="caom"
export WANDB_PROJECT="protein_diffusion"

# Print environment info
echo "Python path: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Starting training..."

# Set Python to unbuffered output
export PYTHONUNBUFFERED=1

# Run training with proper output redirection
python -u main.py \
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
    ++train.strategy=ddp \
    ++train.precision=16 \
    ++train.limit_val_batches=2 \
    ++train.num_sanity_val_steps=1 \
    ++train.accumulate_grad_batches=1 \
    ++model.hidden_size=256 \
    ++model.depth=6 \
    ++model.num_heads=8 \
    ++model.mlp_ratio=4 \
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
    ++dataset.root=/home/caom/AID3/protein-dit/data/protein_train \
    ++data.train_path=/home/caom/AID3/protein-dit/data/protein_train/processed/protein_train_split.pt \
    ++data.val_path=/home/caom/AID3/protein-dit/data/protein_train/processed/protein_val_split.pt \
    ++data.test_path=/home/caom/AID3/protein-dit/data/protein_train/processed/protein_test_split.pt \
    ++dataset.edge_method=knn \
    ++dataset.k=16 \
    > >(tee -a training.log) 2> >(tee -a training.log >&2) 