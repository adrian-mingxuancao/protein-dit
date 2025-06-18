# Protein-DiT: Protein Diffusion Transformer

A standalone implementation of diffusion transformers for protein structure generation, based on the Graph-DiT architecture.

## Overview

This repository contains a complete implementation for training diffusion models on protein structures using transformer architectures. The code is organized as a standalone project that can be easily shared with collaborators.

## Project Structure

```
protein-dit/
├── data/                    # Data directory
│   └── protein_train/       # Training data
├── logs/                    # Training logs
├── outputs/                 # Model outputs
├── wandb/                   # Weights & Biases logs
├── configs/                 # Configuration files
├── datasets/                # Dataset processing
├── diffusion_model/         # Core diffusion model
├── metrics/                 # Training and evaluation metrics
├── utils/                   # Utility functions
├── dataloader/              # Data loading utilities
├── analysis/                # Analysis tools
├── diffusion/               # Diffusion utilities
├── main.py                  # Main training script
├── train.slurm              # SLURM training script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd protein-dit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The data should be placed in the `data/protein_train/` directory. The expected structure is:

```
data/protein_train/
├── processed/
│   ├── protein_train_split.pt
│   ├── protein_val_split.pt
│   └── protein_test_split.pt
└── raw/
    └── [raw protein data files]
```

### Downloading Data Files

Due to GitHub's file size limits, the large data files (`.pt` files) are not included in this repository. You can download them from:

1. **Compressed data file**: `data/protein_train/processed/protein_data.tar.gz` (2.0 GB)
2. Extract the files:
   ```bash
   cd data/protein_train/processed
   tar -xzf protein_data.tar.gz
   ```

Alternatively, you can generate the data files yourself using the provided scripts in the `datasets/` directory.

## Training

### Local Training
```bash
python main.py --config-name=train_config
```

### SLURM Training
```bash
sbatch train.slurm
```

### Configuration

Key configuration parameters can be modified in `configs/train_config.yaml` or passed as command line arguments:

- `train.batch_size`: Batch size for training
- `model.hidden_size`: Hidden dimension of the transformer
- `model.depth`: Number of transformer layers
- `dataset.edge_method`: Edge construction method ('knn' or 'radius')
- `dataset.k`: Number of neighbors for kNN graph construction

## Model Architecture

The model consists of:
- **Transition Model**: Handles the diffusion process transitions
- **Denoiser**: Transformer-based denoising network
- **Noise Schedule**: Manages the noise scheduling during diffusion

## Key Features

- **kNN Graph Construction**: Efficient sparse graph construction using k-nearest neighbors
- **Memory Optimization**: Chunked processing to handle large protein structures
- **Flexible Configuration**: Easy parameter tuning through YAML configs
- **Logging**: Comprehensive logging with Weights & Biases integration

## Usage Examples

### Basic Training
```bash
python main.py \
    ++train.batch_size=32 \
    ++model.hidden_size=256 \
    ++dataset.edge_method=knn \
    ++dataset.k=16
```

### Custom Configuration
```bash
python main.py \
    --config-name=custom_config \
    ++train.max_epochs=200 \
    ++model.depth=8
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information here]
```

## Contact

For questions and issues, please open an issue on GitHub or contact [your-email].

