import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import requests
import tarfile
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import warnings
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import json
import gzip
import pandas as pd

class ProteinNetDataset(Dataset):
    def __init__(self, root='/net/scratch/caom/proteinnet_data', transform=None, pre_transform=None, pre_filter=None):
        """
        Initialize ProteinNet dataset
        Args:
            root: Directory to store ProteinNet data (default: /net/scratch/caom/proteinnet_data)
            transform: Optional transform to be applied on the data
            pre_transform: Optional transform to be applied on the data before saving
            pre_filter: Optional filter to be applied on the data before saving
        """
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data_list = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['training.txt', 'validation.txt', 'testing.txt']
    
    @property
    def processed_file_names(self):
        return ['proteinnet.pt']
    
    def download(self):
        loader = ProteinNetLoader(self.raw_dir)
        loader.download_proteinnet()
    
    def process(self):
        loader = ProteinNetLoader(self.raw_dir)
        data_list = []
        
        # Process each split
        for split in ['training', 'validation', 'testing']:
            try:
                dataset = loader.load_dataset(split)
                data_list.extend(dataset)
            except FileNotFoundError:
                print(f"Warning: {split} data not found, skipping...")
        
        # Save processed data
        torch.save(data_list, self.processed_paths[0])
        
        # Create metadata file
        metadata = {
            "num_graphs": len(data_list),
            "num_features": data_list[0].x.size(1) if data_list else 0,
            "num_classes": 1,  # For protein structure prediction
            "task": "protein_structure",
            "split": {
                "train": len([d for d in data_list if d.split == 'train']),
                "val": len([d for d in data_list if d.split == 'val']),
                "test": len([d for d in data_list if d.split == 'test'])
            }
        }
        
        with open(os.path.join(self.processed_dir, 'proteinnet.meta.json'), 'w') as f:
            json.dump(metadata, f)
    
    def len(self):
        """Return the number of graphs in the dataset."""
        return len(self.data_list)
    
    def get(self, idx):
        """Get a single graph from the dataset."""
        return self.data_list[idx]

class ProteinNetLoader:
    def __init__(self, data_dir='/net/scratch/caom/proteinnet_data', thinning=30):
        """
        Initialize ProteinNet data loader
        Args:
            data_dir: Directory to store ProteinNet data (default: /net/scratch/caom/proteinnet_data)
            thinning: Thinning parameter for ProteinNet (30, 50, 70, 90, 95, 100)
        """
        self.data_dir = data_dir
        self.thinning = thinning
        self.parser = PDBParser(QUIET=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
        
    def download_proteinnet(self, casp_version=12):
        """Download ProteinNet data for specified CASP version"""
        # Use the correct URL from ProteinNet repository
        url = f"https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp{casp_version}.tar.gz"
        print(f"Downloading ProteinNet data from: {url}")
        
        # Download and extract the main archive
        filename = os.path.join(self.data_dir, f"casp{casp_version}.tar.gz")
        
        try:
            # Download file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm(
                desc="Downloading ProteinNet data",
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            
            # Extract file
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(path=self.data_dir)
            
            # Remove tar file
            os.remove(filename)
            
            print("Download and extraction completed successfully!")
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
            if os.path.exists(filename):
                os.remove(filename)
            raise
        except Exception as e:
            print(f"Error processing data: {e}")
            if os.path.exists(filename):
                os.remove(filename)
            raise
    
    def process_proteinnet_entry(self, entry, split):
        """Process a single ProteinNet entry into graph format"""
        # Parse the entry
        lines = entry.strip().split('\n')
        metadata = {}
        current_section = None
        
        for line in lines:
            if line.startswith('#'):
                current_section = line[1:].strip()
                metadata[current_section] = []
            else:
                metadata[current_section].append(line)
        
        # Extract sequence and structure information
        primary = ''.join(metadata['PRIMARY'])
        tertiary = np.array([float(x) for x in ''.join(metadata['TERTIARY']).split()])
        tertiary = tertiary.reshape(-1, 3)
        
        # Create graph data
        data = Data()
        
        # Node features (amino acid types)
        data.x = torch.tensor([ord(c) - ord('A') for c in primary], dtype=torch.long).unsqueeze(1)
        
        # Node positions (CA coordinates)
        data.coords_ca = torch.tensor(tertiary, dtype=torch.float)
        
        # Add additional features
        data.bb_embs = self._compute_backbone_embeddings(data.coords_ca)
        data.side_chain_embs = self._compute_side_chain_embeddings(data.coords_ca)
        
        # Add split information
        data.split = split
        
        return data
    
    def _compute_backbone_embeddings(self, coords):
        """Compute backbone embeddings from CA coordinates"""
        # Compute distances between consecutive CA atoms
        diff = coords[1:] - coords[:-1]
        distances = torch.norm(diff, dim=1)
        
        # Compute angles between consecutive CA atoms
        angles = torch.zeros((len(coords), 3))
        if len(coords) > 2:
            v1 = coords[1:] - coords[:-1]
            v2 = coords[2:] - coords[1:-1]
            cos_angles = torch.sum(v1[:-1] * v2, dim=1) / (torch.norm(v1[:-1], dim=1) * torch.norm(v2, dim=1))
            angles[1:-1, 0] = torch.acos(torch.clamp(cos_angles, -1.0, 1.0))
        
        # Combine features
        features = torch.cat([
            torch.sin(angles),
            torch.cos(angles)
        ], dim=1)
        
        return features
    
    def _compute_side_chain_embeddings(self, coords):
        """Compute side chain embeddings"""
        # For now, use a simplified version based on local structure
        # This can be enhanced with more sophisticated features
        features = torch.zeros((len(coords), 8))
        
        if len(coords) > 2:
            # Compute local curvature
            v1 = coords[1:] - coords[:-1]
            v2 = coords[2:] - coords[1:-1]
            cross = torch.cross(v1[:-1], v2)
            features[1:-1, :3] = cross / (torch.norm(cross, dim=1, keepdim=True) + 1e-6)
            
            # Compute local density
            for i in range(1, len(coords)-1):
                local_coords = coords[max(0, i-2):min(len(coords), i+3)]
                center = torch.mean(local_coords, dim=0)
                features[i, 3:6] = coords[i] - center
                
            # Add distance to center of mass
            com = torch.mean(coords, dim=0)
            features[:, 6:] = coords - com
        
        return features
    
    def load_dataset(self, split='training'):
        """Load ProteinNet dataset for specified split"""
        data_path = os.path.join(self.data_dir, f"{split}.txt")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"ProteinNet {split} data not found. Please download first.")
        
        with open(data_path, 'r') as f:
            entries = f.read().split('\n\n')
        
        dataset = []
        for entry in tqdm(entries, desc=f"Processing {split} data"):
            if entry.strip():
                data = self.process_proteinnet_entry(entry, split)
                dataset.append(data)
        
        return dataset

# Example usage
if __name__ == "__main__":
    # Create dataset with data stored in scratch
    dataset = ProteinNetDataset(root='/net/scratch/caom/proteinnet_data')
    
    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        print(f"Number of node features: {dataset[0].x.size(1)}")
        print(f"Number of edge features: {dataset[0].edge_attr.size(1) if hasattr(dataset[0], 'edge_attr') else 0}")
        print(f"Number of classes: {dataset[0].y.size(1) if hasattr(dataset[0], 'y') else 0}") 