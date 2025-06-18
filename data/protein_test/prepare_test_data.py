import os
import pandas as pd
import shutil

# Source PDB files
pdb_dir = "/net/scratch/caom/test_alphafold/output/reference_hbb/reference_hbb"
pdb_files = [f"ranked_{i}.pdb" for i in range(5)]  # ranked_0.pdb to ranked_4.pdb

# Create raw directory if it doesn't exist
raw_dir = "raw"
os.makedirs(raw_dir, exist_ok=True)

# Copy PDB files
for pdb_file in pdb_files:
    src_path = os.path.join(pdb_dir, pdb_file)
    dst_path = os.path.join(raw_dir, pdb_file)
    shutil.copy2(src_path, dst_path)

# Create metadata CSV
data = []
for pdb_file in pdb_files:
    # For testing, we'll use the same sequence for all files
    # In a real scenario, you would extract this from the PDB
    sequence = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
    
    # Create multiple entries for batch testing
    for i in range(32):  # Create 32 copies for batch testing
        data.append({
            'pdb_file': pdb_file,
            'sequence': sequence,
            'stability': 0.0,  # Placeholder value
            'binding': 0.0,    # Placeholder value
            'fold': 0          # Placeholder value
        })

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv(os.path.join(raw_dir, "hbb_test.csv.gz"), index=False, compression='gzip')

print("Test data preparation complete!") 