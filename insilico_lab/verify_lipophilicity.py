
import pandas as pd
import os
import sys

def verify_dataset(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    try:
        df = pd.read_csv(filepath)
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Check for SMILES and target
        possible_smiles = ['smiles', 'SMILES', 'canonical_smiles']
        possible_target = ['exp', 'logD', 'logP', 'target']
        
        found_smiles = [c for c in df.columns if c in possible_smiles]
        found_target = [c for c in df.columns if c in possible_target]
        
        if not found_smiles:
            print("ERROR: SMILES column not found!")
        else:
            print(f"SMILES column: {found_smiles[0]}")
            
        if not found_target:
            print("ERROR: Target column not found!")
        else:
            print(f"Target column: {found_target[0]}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir)) # d:/chopper/project/insilico_lab/src -> d:/chopper/project/insilico_lab
    # Actually, script is in project root or src... let's assume run from project root or adjust
    # If this script is in project_root, then project_root is current_dir.
    # But I will put this script in d:/chopper/project/insilico_lab/verify_lipophilicity.py
    # So project_root is current_dir.
    
    data_path = 'data/raw/lipophilicity.csv'
    verify_dataset(data_path)
