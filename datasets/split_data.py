import os
import sys
from pathlib import Path
import torch
from torch_geometric.data import Data
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add protein_dit to Python path
current_dir = Path(os.path.realpath(__file__)).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

def split_protein_data():
    # Setup paths
    base_path = Path(os.path.realpath(__file__)).parents[2]  # Go up to protein-dit root
    data_root = base_path / "data" / "protein_train"
    processed_dir = data_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the data
    data_path = processed_dir / "protein_train.pt"
    if not data_path.exists():
        print(f"Error: {data_path} does not exist")
        return
    
    print(f"Loading data from {data_path}")
    data, slices = torch.load(data_path)
    
    # Get number of proteins from slices
    n_proteins = len(slices['x']) - 1  # -1 because slices include start and end indices
    print(f"\nFound {n_proteins} proteins in the dataset")
    
    # Create indices for splitting
    full_idx = list(range(n_proteins))
    train_ratio, valid_ratio, test_ratio = 0.6, 0.2, 0.2
    
    # First split: train + val, test
    train_val_idx, test_idx = train_test_split(
        full_idx, test_size=test_ratio, random_state=42
    )
    
    # Second split: train, val
    train_idx, val_idx = train_test_split(
        train_val_idx, 
        test_size=valid_ratio/(valid_ratio+train_ratio), 
        random_state=42
    )
    
    # Create new Data objects for each split
    def create_split_data(indices):
        split_data = Data()
        split_slices = {}
        
        # Initialize slices
        for key in slices.keys():
            split_slices[key] = [0]  # Start with 0
        
        # Get all start and end indices for this split
        start_indices = torch.tensor([slices['x'][idx] for idx in indices])
        end_indices = torch.tensor([slices['x'][idx + 1] for idx in indices])
        
        # For each feature, create the split data in one go
        for key in data.keys:
            if isinstance(data[key], torch.Tensor):
                # Get all data for this split
                split_data[key] = torch.cat([
                    data[key][start:end] for start, end in zip(start_indices, end_indices)
                ])
        
        # Update slices
        for key in slices.keys():
            split_slices[key] = [0] + torch.cumsum(end_indices - start_indices, dim=0).tolist()
        
        return split_data, split_slices
    
    # Create and save splits
    print("Creating and saving splits...")
    print("Creating train split...")
    train_data, train_slices = create_split_data(train_idx)
    print("Creating validation split...")
    val_data, val_slices = create_split_data(val_idx)
    print("Creating test split...")
    test_data, test_slices = create_split_data(test_idx)
    
    # Save splits
    print("Saving splits...")
    torch.save((train_data, train_slices), processed_dir / "protein_train_split.pt")
    torch.save((val_data, val_slices), processed_dir / "protein_val_split.pt")
    torch.save((test_data, test_slices), processed_dir / "protein_test_split.pt")
    
    # Update metadata with split information
    metadata = {
        "num_graphs": n_proteins,
        "num_features": int(data.x.size(1)),  # Convert to int
        "num_classes": 20,  # Number of amino acids
        "task": "protein_structure",
        "max_nodes": int(max(slices['x'][i+1] - slices['x'][i] for i in range(n_proteins))),  # Convert to int
        "node_features": {
            "amino_acid": int(data.x.size(1)),  # Convert to int
            "side_chain": int(data.side_chain_embs.size(1)),  # Convert to int
            "backbone": int(data.bb_embs.size(1))  # Convert to int
        },
        "edge_features": {
            "distance_bins": 5  # Number of distance bins
        },
        "split": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx)
        }
    }
    
    with open(processed_dir / 'protein_train.meta.json', 'w') as f:
        json.dump(metadata, f)
    
    print("\nData Split Summary:")
    print(f"Total proteins: {n_proteins}")
    print(f"Training set size: {len(train_idx)}")
    print(f"Validation set size: {len(val_idx)}")
    print(f"Test set size: {len(test_idx)}")
    
    # Print shapes of first protein in each split
    print("\nFeature shapes in splits:")
    for split_name, (split_data, split_slices) in [
        ('train', (train_data, train_slices)),
        ('val', (val_data, val_slices)),
        ('test', (test_data, test_slices))
    ]:
        print(f"\n{split_name.capitalize()} split:")
        for key in split_data.keys:
            if isinstance(split_data[key], torch.Tensor):
                print(f"{key}: {split_data[key].shape}")  # Show total shape instead of first protein

if __name__ == "__main__":
    split_protein_data() 