import os
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import *
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

class ProteinVisualization:
    def __init__(self, dataset_infos):
        self.dataset_infos = dataset_infos

    def visualize_by_sequence(self, save_path, sequences, num_samples_to_save):
        """Visualize protein sequences and their properties.
        
        Args:
            save_path: Path to save visualizations
            sequences: List of protein sequences
            num_samples_to_save: Number of samples to save
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save sequences to file
        with open(os.path.join(save_path, 'sequences.txt'), 'w') as f:
            for i, seq in enumerate(sequences[:num_samples_to_save]):
                f.write(f"Sequence {i+1}:\n{seq}\n\n")
        
        # Compute and visualize sequence properties
        self.visualize_sequence_properties(sequences[:num_samples_to_save], save_path)

    def visualize_sequence_properties(self, sequences, save_path):
        """Visualize properties of protein sequences.
        
        Args:
            sequences: List of protein sequences
            save_path: Path to save visualizations
        """
        # Compute amino acid composition
        aa_composition = self.compute_aa_composition(sequences)
        
        # Plot amino acid composition
        plt.figure(figsize=(12, 6))
        plt.bar(aa_composition.keys(), aa_composition.values())
        plt.title('Amino Acid Composition')
        plt.xlabel('Amino Acid')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'aa_composition.png'))
        plt.close()
        
        # Compute and plot sequence length distribution
        lengths = [len(seq) for seq in sequences]
        plt.figure(figsize=(8, 6))
        plt.hist(lengths, bins=20)
        plt.title('Sequence Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'length_distribution.png'))
        plt.close()

    def compute_aa_composition(self, sequences):
        """Compute amino acid composition of sequences."""
        aa_counts = {}
        total_aas = 0
        
        for seq in sequences:
            for aa in seq:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
                total_aas += 1
        
        # Convert to frequencies
        aa_freqs = {aa: count/total_aas for aa, count in aa_counts.items()}
        return aa_freqs

    def visualize_structure(self, pdb_path, save_path):
        """Visualize protein structure.
        
        Args:
            pdb_path: Path to PDB file
            save_path: Path to save visualization
        """
        try:
            # Load structure
            parser = PDBParser()
            structure = parser.get_structure('protein', pdb_path)
            
            # Create 3D visualization
            plt.figure(figsize=(10, 10))
            ax = plt.axes(projection='3d')
            
            # Plot CA atoms
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            ca = residue['CA']
                            ax.scatter(ca.get_coord()[0], ca.get_coord()[1], ca.get_coord()[2])
            
            plt.title('Protein Structure')
            plt.savefig(os.path.join(save_path, 'structure.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing structure: {str(e)}")

    def visualize_secondary_structure(self, pdb_path, save_path):
        """Visualize secondary structure of protein.
        
        Args:
            pdb_path: Path to PDB file
            save_path: Path to save visualization
        """
        try:
            # Get secondary structure using DSSP
            dssp = dssp_dict_from_pdb_file(pdb_path)
            
            # Count secondary structure elements
            ss_counts = {'H': 0, 'E': 0, 'C': 0}  # H: helix, E: sheet, C: coil
            for chain in dssp.values():
                for ss in chain:
                    ss_counts[ss[1]] += 1
            
            # Plot secondary structure distribution
            plt.figure(figsize=(8, 6))
            plt.bar(ss_counts.keys(), ss_counts.values())
            plt.title('Secondary Structure Distribution')
            plt.xlabel('Secondary Structure')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'secondary_structure.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing secondary structure: {str(e)}") 