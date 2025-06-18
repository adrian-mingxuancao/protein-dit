import os
import gzip
import shutil
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
from tqdm import tqdm
from typing import Optional, List, Tuple
import logging
import requests
import tarfile
import json
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .pronet_aa import (
    get_atom_pos,
    compute_side_chain_embs,
    compute_bb_embs,
    pronet_graph_gen
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProteinNetDataset(Dataset):
    """Dataset for loading ProteinNet data in graph format."""
    
    def __init__(
        self,
        root: str = "/net/scratch/caom/proteinnet_data",
        split: str = "train",
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
        pre_filter: Optional[callable] = None,
        max_length: int = 1000,
        casp_version: int = 12,
    ):
        """
        Args:
            root (str): Root directory where the dataset should be saved (default: /net/scratch/caom/proteinnet_data)
            split (str): One of 'train', 'valid', or 'test'
            transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version
            pre_transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version
            pre_filter (callable, optional): A function that takes in a Data object and returns a boolean value
            max_length (int): Maximum protein length to include in dataset
            casp_version (int): CASP version to use (default: 12)
        """
        self.split = split
        self.max_length = max_length
        self.casp_version = casp_version
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self) -> List[str]:
        """Returns list of raw file names."""
        return [f"casp{self.casp_version}.tar.gz"]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Returns list of processed file names."""
        return [f"data_{self.split}_{i}.pt" for i in range(len(self.processed_paths))]
    
    def download(self):
        """Download PDB files for existing ProteinNet data."""
        # Create PDB directory
        pdb_dir = os.path.join(self.raw_dir, 'pdb')
        os.makedirs(pdb_dir, exist_ok=True)
        
        # Get the correct split file path
        split_files = {
            'train': [f'training_{i}' for i in [30, 70, 90, 95, 100]],
            'valid': ['validation'],
            'test': ['testing']
        }
        
        # Process each split file
        for split_file in split_files.get(self.split, []):
            file_path = os.path.join(self.raw_dir, split_file)
            if not os.path.exists(file_path):
                logger.warning(f"Split file not found: {file_path}")
                continue
                
            logger.info(f"Processing split file: {file_path}")
            with open(file_path, 'r') as f:
                entries = f.read().split('\n\n')
            
            for entry in tqdm(entries, desc=f"Processing {split_file}"):
                if not entry.strip():
                    continue
                    
                try:
                    # Parse entry to get PDB ID
                    lines = entry.strip().split('\n')
                    pdb_id = lines[0].split()[1]
                    
                    # Try to download from AlphaFold first
                    pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
                    if not os.path.exists(pdb_file):
                        # Try AlphaFold first
                        af_url = f"https://alphafold.ebi.ac.uk/files/AF-{pdb_id}-F1-model_v4.pdb"
                        try:
                            response = requests.get(af_url)
                            response.raise_for_status()
                            with open(pdb_file, 'wb') as f:
                                f.write(response.content)
                            logger.info(f"Downloaded AlphaFold structure for {pdb_id}")
                            continue
                        except requests.exceptions.RequestException:
                            logger.warning(f"AlphaFold structure not found for {pdb_id}, trying RCSB PDB...")
                        
                        # If AlphaFold fails, try RCSB PDB
                        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                        try:
                            response = requests.get(pdb_url)
                            response.raise_for_status()
                            with open(pdb_file, 'wb') as f:
                                f.write(response.content)
                            logger.info(f"Downloaded RCSB PDB structure for {pdb_id}")
                        except requests.exceptions.RequestException as e:
                            logger.warning(f"Failed to download structure for {pdb_id}: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error processing entry in {split_file}: {str(e)}")
                    continue
                    
        logger.info("PDB download complete!")
    
    def process(self):
        """Process raw data into graph format."""
        parser = PDBParser(QUIET=True)
        
        # Get list of PDB files in the raw directory
        pdb_dir = os.path.join(self.raw_dir, 'pdb')
        pdb_files = []
        for file in os.listdir(pdb_dir):
            if file.endswith('.pdb'):
                pdb_files.append(os.path.join(pdb_dir, file))
        
        logger.info(f"Found {len(pdb_files)} PDB files to process")
        
        processed_count = 0
        for i, pdb_file in enumerate(tqdm(pdb_files, desc="Processing proteins")):
            try:
                # Generate graph from PDB file
                data, seq = pronet_graph_gen(
                    pdb_file,
                    parser
                )
                
                # Skip if protein is too long
                if len(data.x) > self.max_length:
                    continue
                    
                # Add sequence information
                data.sequence = seq
                
                # Add protein ID
                data.protein_id = os.path.basename(pdb_file).replace('.pdb', '')
                
                # Add split information
                data.split = self.split
                
                # Save processed data
                torch.save(data, os.path.join(self.processed_dir, f'data_{self.split}_{processed_count}.pt'))
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing {pdb_file}: {str(e)}")
                continue
                
        logger.info(f"Successfully processed {processed_count} proteins")
                
    def len(self) -> int:
        """Return number of proteins in dataset."""
        return len(self.processed_file_names)
        
    def get(self, idx: int) -> Data:
        """Get protein graph by index."""
        data = torch.load(os.path.join(self.processed_dir, f'data_{self.split}_{idx}.pt'))
        return data 