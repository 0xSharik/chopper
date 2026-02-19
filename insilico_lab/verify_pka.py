
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
        possible_smiles = ['SMILES', 'smiles', 'structure', 'canonical_smiles']
        possible_pka = ['pKa', 'pka', 'PKA', 'value', 'pKa_value']
        
        found_smiles = [c for c in df.columns if c in possible_smiles]
        found_pka = [c for c in df.columns if c in possible_pka]
        
        if not found_smiles:
            print("ERROR: SMILES column not found!")
        else:
            print(f"SMILES column: {found_smiles[0]}")
            
        if not found_pka:
            print("ERROR: pKa column not found!")
        else:
            print(f"pKa column: {found_pka[0]}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    data_path = 'data/raw/pka.csv'
    verify_dataset(data_path)
