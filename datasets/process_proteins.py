import os
import sys
import gzip
import shutil
import json
from pathlib import Path
import torch
from Bio.PDB import PDBParser
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add protein_dit to Python path
current_dir = Path(os.path.realpath(__file__)).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from dataloader.pronet_aa import pronet_graph_gen

class ProteinProcessor:
    """A unified processor for both AlphaFold and regular PDB files."""
    
    def __init__(self, data_root, source_type='alphafold', is_gzipped=True):
        """
        Initialize the protein processor.
        
        Args:
            data_root (str): Root directory for the dataset
            source_type (str): Type of source data ('alphafold' or 'pdb')
            is_gzipped (bool): Whether the input files are gzipped
        """
        self.data_root = Path(data_root)
        self.source_type = source_type
        self.is_gzipped = is_gzipped
        self.processed_dir = self.data_root / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup source directory based on type
        if source_type == 'alphafold':
            self.source_dir = Path("/net/scratch/caom/proteinnet_data/raw/alphafold/pdb")
        else:  # regular pdb
            self.source_dir = Path("/net/scratch/caom/test_alphafold/output/reference_hbb/reference_hbb")
    
    def process_file(self, pdb_path, parser):
        """Process a single PDB file."""
        try:
            # Handle gzipped files
            if self.is_gzipped:
                temp_path = Path("/tmp") / pdb_path.stem
                with gzip.open(pdb_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                pdb_path = temp_path
            
            # Parse PDB and convert to graph
            data, seqs = pronet_graph_gen(str(pdb_path), parser)
            
            # Add protein ID and sequence
            data.protein_id = pdb_path.stem
            data.sequence = seqs
            
            # Clean up temporary file if needed
            if self.is_gzipped:
                temp_path.unlink()
            
            return data
            
        except Exception as e:
            print(f"Error processing {pdb_path}: {str(e)}")
            return None
    
    def process_dataset(self):
        """Process all PDB files in the source directory."""
        print(f"Processing {self.source_type} PDB files...")
        data_list = []
        parser = PDBParser(QUIET=True)
        
        # Get all PDB files
        if self.source_type == 'alphafold':
            pdb_files = list(self.source_dir.glob("*.pdb.gz"))
        else:
            pdb_files = [self.source_dir / f"ranked_{i}.pdb" for i in range(5)]
        
        print(f"Found {len(pdb_files)} PDB files")
        
        # Process files with progress bar
        pbar = tqdm(pdb_files, desc="Processing proteins", unit="protein")
        successful = 0
        failed = 0
        
        for pdb_path in pbar:
            data = self.process_file(pdb_path, parser)
            if data is not None:
                data_list.append(data)
                successful += 1
            else:
                failed += 1
            
            pbar.set_postfix({
                'successful': successful,
                'failed': failed,
                'total': len(pdb_files)
            })
        
        if data_list:
            # Save processed data
            output_name = "protein_train.pt" if self.source_type == 'alphafold' else "hbb_test.pt"
            save_path = self.processed_dir / output_name
            torch.save(data_list, save_path)
            
            # Create metadata
            metadata = {
                "num_graphs": len(data_list),
                "num_features": data_list[0].x.size(1),
                "num_classes": 20,  # Number of amino acids
                "task": "protein_structure",
                "max_nodes": max(data.num_nodes for data in data_list),
                "node_features": {
                    "amino_acid": data_list[0].x.size(1),
                    "side_chain": data_list[0].side_chain_embs.size(1),
                    "backbone": data_list[0].bb_embs.size(1)
                },
                "edge_features": {
                    "distance_bins": 5  # Number of distance bins
                }
            }
            
            with open(self.processed_dir / f"{output_name[:-3]}.meta.json", 'w') as f:
                json.dump(metadata, f)
            
            print("\nProcessed Data Summary:")
            print(f"Total proteins: {len(data_list)}")
            print(f"Successfully processed: {successful}")
            print(f"Failed to process: {failed}")
            print(f"Node features shape: {data_list[0].x.shape}")
            print(f"Side chain embeddings shape: {data_list[0].side_chain_embs.shape}")
            print(f"Backbone embeddings shape: {data_list[0].bb_embs.shape}")
            print(f"CA coordinates shape: {data_list[0].coords_ca.shape}")
        else:
            print("No data was processed successfully.")

def main():
    # Get base path
    base_path = Path(os.path.realpath(__file__)).parents[2]  # Go up to protein-dit root
    
    # Process AlphaFold dataset
    print("\nProcessing AlphaFold dataset...")
    alphafold_processor = ProteinProcessor(
        data_root=base_path / "data" / "protein_train",
        source_type='alphafold',
        is_gzipped=True
    )
    alphafold_processor.process_dataset()
    
    # Process test PDB dataset
    print("\nProcessing test PDB dataset...")
    pdb_processor = ProteinProcessor(
        data_root=base_path / "data" / "protein_test",
        source_type='pdb',
        is_gzipped=False
    )
    pdb_processor.process_dataset()

if __name__ == "__main__":
    main() 