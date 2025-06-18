# Protein Design with Graph-DiT

## Project Structure

## 1. Data Preprocessing
- [x] Raw PDB files from AlphaFold
- [x] Convert to PyG format
  - [x] Fix import issues
  - [x] Process PDBs using ProNet's approach
  - [x] Validate output format
- [x] Data validation
  - [x] Check node features
  - [x] Check edge features
  - [x] Verify embeddings

## 2. Model Integration
- [x] Data compatibility check
  - [x] Verify input shapes
  - [x] Test forward pass
- [x] Model architecture
  - [x] Graph-DiT adaptation
  - [x] Protein-specific layers
- [x] Integration testing
  - [x] Small batch testing
  - [x] Full pipeline test

## 3. Training Pipeline
- [ ] Training setup
  - [ ] Data loading
  - [ ] Model configuration
  - [ ] Optimizer setup
- [ ] Training loop
  - [ ] Loss functions
  - [ ] Validation
  - [ ] Checkpointing
- [ ] Evaluation
  - [ ] Metrics
  - [ ] Visualization

## Current Status

### Data Preprocessing ✅
- Raw PDB files are available from AlphaFold (ranked_0.pdb to ranked_4.pdb)
- Data structure is set up in `Graph-DiT/data/protein_test/`
- Processing scripts are in place:
  - `process_pdbs.py`: Converts PDBs to PyG format using ProNet's approach
  - Data is saved in `Graph-DiT/data/protein_test/processed/hbb_test.pt`
- Verified data shapes and features:
  - Node features: [147, 1]
  - Side chain embeddings: [147, 8]
  - Backbone embeddings: [147, 6]
  - CA coordinates: [147, 3]

### Model Integration ✅
- Successfully adapted Graph-DiT for protein data:
  - Modified SELayer and ProteinAttention for proper tensor handling
  - Implemented ProteinOutLayer for amino acid and edge feature outputs
  - Created ProteinDenoiser with sequence-aware processing
- Verified data compatibility:
  - Input shapes: x[2, 147, 20], e[2, 147, 147, 5], p[2, 147, 8]
  - Output shapes: X[2, 147, 20], E[2, 147, 147, 5]
  - Proper masking and modulation throughout the network
- Completed integration testing:
  - Successful forward pass with small batches
  - Correct tensor operations in attention layers
  - Proper handling of protein-specific features

### Training Pipeline ⏳
- Basic training configuration is set up
- Data loading and model architecture are ready
- Training loop needs to be implemented

## Next Steps

1. **Analyze Graph-DiT's model architecture:**
   - Study how it handles molecular data
   - Identify necessary modifications for proteins

2. **Adapt model for proteins:**
   - Modify node features for amino acids
   - Update edge features for protein contacts
   - Integrate protein-specific embeddings

3. **Set up training pipeline:**
   - Implement data loading
   - Define loss functions
   - Add validation metrics

## File Structure
```
protein_dit/
├── datasets/
│   ├── process_pdbs.py
│   └── test_processed.py
├── dataloader/
│   └── pronet_aa.py
├── utils/
│   ├── pronet_utils.py
│   └── helper.py
├── configs/
│   └── train_config.yaml
└── main.py
```

## Dependencies
- PyTorch
- PyTorch Geometric
- BioPython
- NumPy
- Pandas
- scikit-learn
- tqdm
- hydra-core
- pytorch-lightning 