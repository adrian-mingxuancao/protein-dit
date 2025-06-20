#!/usr/bin/env python3
"""
Test script to verify model parameters without running full training.
This can be run on the login node to check configuration.
"""

import os
import sys
import torch
import hydra
from omegaconf import DictConfig

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_config():
    """Test the model configuration and print parameter count."""
    
    # Initialize hydra
    hydra.initialize(version_base="1.1", config_path="configs")
    
    # Load config
    cfg = hydra.compose(config_name="train_config")
    
    print("=== Model Configuration ===")
    print(f"hidden_size: {cfg.model.hidden_size}")
    print(f"depth: {cfg.model.depth}")
    print(f"num_heads: {cfg.model.num_heads}")
    print(f"mlp_ratio: {cfg.model.mlp_ratio}")
    print(f"diffusion_steps: {cfg.model.diffusion_steps}")
    print()
    
    print("=== Training Configuration ===")
    print(f"batch_size: {cfg.train.batch_size}")
    print(f"max_epochs: {cfg.train.max_epochs}")
    print(f"devices: {cfg.train.devices}")
    print(f"precision: {cfg.train.precision}")
    print()
    
    print("=== Data Configuration ===")
    print(f"train_path: {cfg.data.train_path}")
    print(f"val_path: {cfg.data.val_path}")
    print(f"test_path: {cfg.data.test_path}")
    print()
    
    # Calculate expected parameter count (approximate)
    # This is a rough estimate based on the transformer architecture
    hidden_size = cfg.model.hidden_size
    depth = cfg.model.depth
    num_heads = cfg.model.num_heads
    mlp_ratio = cfg.model.mlp_ratio
    
    # Approximate parameter calculation for transformer layers
    # Each layer has: attention + MLP + layer norms
    attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
    mlp_params = 2 * hidden_size * (hidden_size * mlp_ratio)  # MLP layers
    layer_norm_params = 2 * hidden_size  # 2 layer norms per layer
    
    params_per_layer = attention_params + mlp_params + layer_norm_params
    total_transformer_params = params_per_layer * depth
    
    # Add embedding and output layers
    # Assuming input/output dimensions
    embedding_params = 20 * hidden_size  # 20 amino acid types
    output_params = hidden_size * 20  # Output projection
    
    total_params = total_transformer_params + embedding_params + output_params
    
    print("=== Expected Model Parameters ===")
    print(f"Parameters per layer: {params_per_layer:,}")
    print(f"Total transformer parameters: {total_transformer_params:,}")
    print(f"Total model parameters (approx): {total_params:,}")
    print(f"Model size (approx): {total_params * 4 / 1024 / 1024:.1f} MB")
    print()
    
    print("=== Configuration Summary ===")
    if cfg.model.hidden_size >= 512 and cfg.model.depth >= 12 and cfg.model.num_heads >= 16:
        print("✅ Using LARGE model configuration")
        print("   - This should give ~50-100M+ parameters")
    else:
        print("❌ Using SMALL model configuration")
        print("   - This will give ~10-20M parameters")
    
    print(f"\nConfig file: {os.path.abspath('configs/train_config.yaml')}")
    print("Ready to submit training job!")

if __name__ == "__main__":
    test_model_config() 