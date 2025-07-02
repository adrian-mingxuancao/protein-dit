#!/bin/bash
#SBATCH -p general
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2
#SBATCH -o protein_resume_fixed_%j.out
#SBATCH -e protein_resume_fixed_%j.err

# Exit on error
set -e

# Initialize conda
eval "$(~/miniconda3/bin/conda shell.bash hook)" || { echo "Failed to initialize conda"; exit 1; }
conda activate graph_md4 || { echo "Failed to activate conda environment"; exit 1; }

# Set working directory
cd /home/caom/AID3/protein-dit || { echo "Failed to change to working directory"; exit 1; }

# Set a specific port for DDP to avoid conflicts
export MASTER_PORT=29500

# Add protein-dit to Python path
export PYTHONPATH=/home/caom/AID3/protein-dit:$PYTHONPATH

# Set wandb environment variables
export WANDB_API_KEY="264f2fc6de81df8451a22c5f50c0d5adca1ae0f7"
export WANDB_ENTITY="caom"
export WANDB_PROJECT="protein_diffusion"

# Set Python to unbuffered output
export PYTHONUNBUFFERED=1

# Print environment info
echo "Python path: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Starting resumed training with NLL fix from epoch 36 to 100..."

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --list-gpus || echo "nvidia-smi not available"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs requested: 2"

# Check if the checkpoint exists
CHECKPOINT_PATH="/home/caom/AID3/protein-dit/outputs/2025-06-24/11-57-51/logs/protein_diffusion/checkpoints/epoch=36-val_NLL=0.00.ckpt"
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "✅ Checkpoint found: $CHECKPOINT_PATH"
else
    echo "❌ Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check if the config file exists
CONFIG_PATH="configs/train_config.yaml"
if [ -f "$CONFIG_PATH" ]; then
    echo "✅ Config file found: $CONFIG_PATH"
else
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG_PATH"
echo "Resume from: $CHECKPOINT_PATH"
echo "=========================================="

# Run the main training script with our fixed config
echo "Starting training with main.py and fixed config..."
srun python main.py > >(tee -a training_resume_fixed.log) 2> >(tee -a training_resume_fixed.log >&2)

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # List the checkpoints that were created
    echo "📁 Checkpoints created:"
    find outputs -name "*.ckpt" -type f | sort
    
    # Check for test checkpoints at epochs 50, 75, 100
    echo "🧪 Looking for test checkpoints:"
    for epoch in 50 75 100; do
        test_checkpoint=$(find outputs -name "*epoch=${epoch}*" -type f | head -1)
        if [ -n "$test_checkpoint" ]; then
            echo "✅ Test checkpoint for epoch $epoch: $test_checkpoint"
        else
            echo "❌ No test checkpoint found for epoch $epoch"
        fi
    done
    
else
    echo "❌ Training failed!"
    exit 1
fi

echo "🎉 Job completed!" 