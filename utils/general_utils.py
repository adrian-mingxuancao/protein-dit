import os
import torch
import numpy as np
from pathlib import Path

def get_project_root():
    """Get the root directory of the project."""
    return Path(os.path.realpath(__file__)).parent.parent

def setup_paths():
    """Setup necessary paths for the project."""
    root = get_project_root()
    return {
        'data': os.path.join(root, 'data'),
        'checkpoints': os.path.join(root, 'checkpoints'),
        'configs': os.path.join(root, 'configs')
    }

def ensure_dir(directory):
    """Ensure a directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 