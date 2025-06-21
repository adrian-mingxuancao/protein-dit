import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from utils.graph_utils import to_dense, create_fully_connected_edges

class DatasetInfo:
    def __init__(self, train_dataset):
        # Get input dimensions from the first protein
        first_protein = train_dataset[0]
        
        # Set input dimensions
        self.input_dims = {
            'X': 20,  # Number of amino acids
            'E': 5,   # Number of edge types (distance bins)
            'P': 20,  # Number of amino acids for sequence
            'y': 3    # Number of target classes
        }
        
        # Set output dimensions (same as input for this case)
        self.output_dims = {
            'X': 20,  # Number of amino acids
            'E': 5,   # Number of edge types
            'P': 20,  # Number of sequence positions
            'y': 3    # Number of target classes
        }
        
        # Set maximum number of nodes from slices
        print("Computing maximum number of nodes...")
        self.max_n_nodes = max(
            train_dataset.slices['x'][i+1] - train_dataset.slices['x'][i]
            for i in range(len(train_dataset.slices['x'])-1)
        )
        print(f"Maximum number of nodes: {self.max_n_nodes}")
        
        # Initialize transition matrices that will be learned during training
        self.transition_xe = torch.zeros(20, 5)  # Q_xe: amino acid to edge transitions [20, 5]
        self.transition_ex = torch.zeros(5, 20)  # Q_ex: edge to amino acid transitions [5, 20]
        
        # Try to load pre-computed metadata
        metadata_path = os.path.join(os.path.dirname(train_dataset.data_path), 'protein_metadata.json')
        if os.path.exists(metadata_path):
            print("Loading pre-computed metadata...")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load distributions from metadata
            self.node_types = torch.tensor(metadata['node_types'])
            self.edge_types = torch.tensor(metadata['edge_types'])
            self.protein_types = torch.tensor(metadata['protein_types'])
            self.transition_E = torch.tensor(metadata['transition_E'])
            self.active_index = torch.tensor(metadata['active_index'])
            
            # Set position types (same as node types for amino acids)
            self.position_types = self.node_types.clone()
            
            print(f"Loaded metadata:")
            print(f"transition_E shape: {self.transition_E.shape}")  # Should be [20, 20]
        else:
            print("Computing and saving metadata...")
            # Initialize distributions
            self.node_types = torch.zeros(20)  # Amino acid frequencies
            self.edge_types = torch.zeros(5)   # Edge type frequencies
            self.protein_types = torch.zeros(20)  # Sequence position frequencies
            
            # Initialize transition matrices
            self.transition_E = torch.zeros(20, 20)  # Q_ee: edge-to-edge transitions [20, 20]
            
            # Compute distributions and transitions from training data with progress tracking
            total_proteins = len(train_dataset)
            for i, data in enumerate(train_dataset):
                if i % 100 == 0:  # Print progress every 100 proteins
                    print(f"Processing protein {i+1}/{total_proteins} ({(i+1)/total_proteins*100:.1f}%)")
                
                # Convert one-hot encoded amino acids to class indices
                x = data.x.argmax(dim=1)  # Shape: [num_nodes]
                self.node_types += torch.bincount(x, minlength=20)
                
                # For edge types, we'll use a uniform distribution initially
                self.edge_types += torch.ones(5)
                
                # For protein types, we'll use a uniform distribution initially
                self.protein_types += torch.ones(20)
            
                # Get edge information
                edge_index = data.edge_index  # [2, E]
                edge_attr = data.edge_attr  # Edge features [E, 1]
                
                # Get source and target nodes for each edge
                src_nodes = edge_index[0]
                dst_nodes = edge_index[1]
                
                # Get amino acid types
                src_types = x[src_nodes].long()  # [E]
                dst_types = x[dst_nodes].long()  # [E]
                
                # Update transition matrix
                self.transition_E[src_types, dst_types] += 1
                self.transition_E[dst_types, src_types] += 1
            
            print("\nNormalizing distributions and matrices...")
            # Normalize distributions
            self.node_types = self.node_types / self.node_types.sum()
            self.edge_types = self.edge_types / self.edge_types.sum()
            self.protein_types = self.protein_types / self.protein_types.sum()
            
            # Normalize transition matrix
            row_sums_E = self.transition_E.sum(dim=1, keepdim=True)
            self.transition_E = self.transition_E / (row_sums_E + 1e-8)
                
            # Set position types (same as node types for amino acids)
            self.position_types = self.node_types.clone()
            
            # Set active indices (all amino acids are active)
            self.active_index = torch.arange(20)
            
            # Save metadata for future use
            metadata = {
                'node_types': self.node_types.tolist(),
                'edge_types': self.edge_types.tolist(),
                'protein_types': self.protein_types.tolist(),
                'transition_E': self.transition_E.tolist(),
                'active_index': self.active_index.tolist()
            }
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        # Set task type
        self.task_type = 'protein_design'
        
        print("Dataset info initialization complete!")

    def convert_node_features_to_sequence(self, node_features):
        """Convert one-hot encoded node features to amino acid sequence.
        
        Args:
            node_features: Tensor of shape [num_nodes, num_classes] with one-hot encoded amino acids
            
        Returns:
            String representation of the amino acid sequence
        """
        # Define amino acid alphabet based on pronet_prepoc encoding
        # The model uses Xdim=20, but original data had 26 classes
        # We map the rare/ambiguous amino acids to standard ones
        amino_acids = [
            'H',  # 0: HIS/HID/HIE/HIP
            'D',  # 1: ASP/ASH
            'R',  # 2: ARG/ARN
            'F',  # 3: PHE
            'A',  # 4: ALA
            'C',  # 5: CYS/CYX
            'G',  # 6: GLY
            'Q',  # 7: GLN
            'E',  # 8: GLU/GLH
            'K',  # 9: LYS/LYN
            'L',  # 10: LEU
            'M',  # 11: MET
            'N',  # 12: ASN
            'S',  # 13: SER
            'Y',  # 14: TYR
            'T',  # 15: THR
            'I',  # 16: ILE
            'W',  # 17: TRP
            'P',  # 18: PRO
            'V',  # 19: VAL (also maps rare amino acids >= 20)
        ]
        
        # Convert one-hot to indices
        if node_features.dim() == 2:
            # One-hot encoded features [num_nodes, num_classes]
            indices = torch.argmax(node_features, dim=1)
        else:
            # Already indices [num_nodes]
            indices = node_features
            
        # Convert tensor to list and flatten if needed
        indices_list = indices.tolist()
        
        # Handle both single samples and batched samples
        if isinstance(indices_list[0], list):
            # Batched sample: flatten the list
            indices_flat = []
            for sublist in indices_list:
                if isinstance(sublist, list):
                    indices_flat.extend(sublist)
                else:
                    indices_flat.append(sublist)
            indices_list = indices_flat
        
        # Convert indices to amino acid sequence
        # Note: Since the model uses 20 classes, any original index >= 20 
        # would have been clamped to 19 during one-hot encoding
        sequence = ''.join([amino_acids[idx] if idx < len(amino_acids) else 'X' for idx in indices_list])
        
        return sequence

class ProteinDataset(InMemoryDataset):
    def __init__(self, data_path: str, transform=None):
        """
        Args:
            data_path: Path to the pre-processed PyG data file
            transform: Optional transform to be applied on the data
        """
        print(f"Initializing ProteinDataset with data_path: {data_path}")
        self.data_path = data_path
        super().__init__(transform=transform)
        
        try:
            print("Loading data file...")
            # Load data and slices
            self._data, self.slices = torch.load(data_path)
            print("Data file loaded successfully")
            
            # Get number of proteins from slices
            self.num_proteins = len(self.slices['x']) - 1
            print(f"Number of proteins found: {self.num_proteins}")
            
            print("\nDataset info:")
            print(f"Number of proteins: {self.num_proteins}")
            
            # Print data attributes for first protein as example
            if self.num_proteins > 0:
                print("\nData attributes and shapes (first protein):")
                try:
                    first_protein = self[0]
                    for key in first_protein.keys:
                        value = getattr(first_protein, key)
                        if isinstance(value, torch.Tensor):
                            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
                        else:
                            print(f"{key}: type {type(value)}")
                except Exception as e:
                    print(f"Error accessing first protein: {str(e)}")
                    raise
        except Exception as e:
            print(f"Error in ProteinDataset initialization: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    @property
    def processed_file_names(self):
        return [os.path.basename(self.data_path)]

    def len(self):
        return self.num_proteins

    def get(self, idx):
        # Get start and end indices for this protein
        start_idx = self.slices['x'][idx]
        end_idx = self.slices['x'][idx + 1]
        
        # Create a new Data object for this protein
        protein_data = Data()
        
        # Copy the data for this protein
        for key in self._data.keys:
            if isinstance(self._data[key], torch.Tensor):
                protein_data[key] = self._data[key][start_idx:end_idx]
            else:
                protein_data[key] = self._data[key]
        
        # Add node_mask if not present
        if not hasattr(protein_data, 'node_mask'):
            protein_data.node_mask = torch.ones(protein_data.num_nodes, dtype=torch.bool)
        
        # Create edge index using kNN
        n_nodes = protein_data.num_nodes
        edge_index = create_fully_connected_edges(
            n_nodes=n_nodes,
            batch_size=1,
            coords=protein_data.coords_ca,  # Pass coordinates for kNN
            edge_method='knn',
            k=16  # Use k=16 as specified in config
        )
        
        # Calculate distances between connected nodes for edge attributes
        if edge_index.shape[1] > 0:  # Only if we have edges
            src_coords = protein_data.coords_ca[edge_index[0]]
            dst_coords = protein_data.coords_ca[edge_index[1]]
            distances = torch.norm(src_coords - dst_coords, dim=1)
            
            # Bin distances into 5 categories: <5Å, 5-10Å, 10-15Å, 15-20Å, >20Å
            edge_attr = torch.zeros(edge_index.shape[1], dtype=torch.long)
            edge_attr[distances < 5.0] = 0
            edge_attr[(distances >= 5.0) & (distances < 10.0)] = 1
            edge_attr[(distances >= 10.0) & (distances < 15.0)] = 2
            edge_attr[(distances >= 15.0) & (distances < 20.0)] = 3
            edge_attr[distances >= 20.0] = 4
        else:
            edge_attr = torch.zeros(0, dtype=torch.long)
        
        # Add edge information
        protein_data.edge_index = edge_index
        protein_data.edge_attr = edge_attr
        
        return protein_data

class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.dataset_info = None
        self.training_iterations = None
        
    def prepare_data(self):
        """Load and prepare the datasets."""
        print("Starting prepare_data...")
        
        # Load training data
        print(f"Loading train data from {self.cfg.data.train_path}")
        self.train_dataset = ProteinDataset(
            data_path=self.cfg.data.train_path,
            transform=None
        )
        
        # Load validation data
        print(f"Loading validation data from {self.cfg.data.val_path}")
        self.val_dataset = ProteinDataset(
            data_path=self.cfg.data.val_path,
            transform=None
        )
        
        # Load test data
        print(f"Loading test data from {self.cfg.data.test_path}")
        self.test_dataset = ProteinDataset(
            data_path=self.cfg.data.test_path,
            transform=None
        )
        
        # Initialize dataset info
        print("Initializing dataset info...")
        self.dataset_info = DatasetInfo(self.train_dataset)
        
        # Calculate training iterations
        self.training_iterations = len(self.train_dataset) // self.cfg.train.batch_size
        print(f"Training iterations per epoch: {self.training_iterations}")
        
        # Create dataloaders
        print("Creating dataloaders...")
        
        # Use small batch sizes for testing, but respect the config
        effective_batch_size = self.cfg.train.batch_size
        actual_batch_size = min(effective_batch_size, 2)  # Cap at 2 for testing, can increase later
        
        print(f"[DEBUG] Using batch_size={actual_batch_size} (effective={effective_batch_size} with accumulation)", flush=True)
        
        # Use max_n_nodes from dataset info (already computed as 2699)
        max_nodes = self.dataset_info.max_n_nodes
        print(f"[DEBUG] Using max_nodes from dataset info: {max_nodes}", flush=True)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=actual_batch_size,  # Use config batch size (capped)
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=actual_batch_size,  # Use same batch size for validation
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=actual_batch_size,  # Use config batch size (capped)
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )
        
        print("Data preparation complete!")
        
    def setup(self, stage=None):
        """Setup is called from every process across all the nodes."""
        # We don't need to do anything here since we've already prepared the data
        # in prepare_data(). This method is required by Lightning but can be empty
        # if we've done all the setup in prepare_data().
        pass
        
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
        
    def test_dataloader(self):
        return self.test_loader

    def example_batch(self):
        """Return an example batch for model initialization"""
        return next(iter(self.val_dataloader()))

    def get_train_sequences(self):
        """Return training and reference sequences for metrics computation"""
        train_sequences = []
        reference_sequences = []
        
        for data in self.train_dataset:
            if hasattr(data, 'sequence'):
                train_sequences.append(data.sequence)
            if hasattr(data, 'reference_sequence'):
                reference_sequences.append(data.reference_sequence)
        
        return train_sequences, reference_sequences 