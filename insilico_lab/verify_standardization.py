
import sys
import os
import pandas as pd
import random
from rdkit import Chem

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_pipeline.load_data import load_data
from data_pipeline.standardize import standardize_smiles

def run_verification():
    print("Loading data...")
    try:
        df = load_data('data/raw/esol.csv')
    except Exception as e:
        print(f"CRITICAL: Failed to load data: {e}")
        return

    total_samples = 10
    print(f"\nRandomly selecting {total_samples} samples for verification...")
    
    # Random sample
    sample_df = df.sample(n=total_samples, random_state=42) # Fixed seed for reproducibility, or remove for true random
    
    failures = 0
    results = []

    print("-" * 80)
    print(f"{'Original SMILES (truncated)':<35} | {'Standardized SMILES (truncated)':<35} | {'Status'}")
    print("-" * 80)

    for i, row in sample_df.iterrows():
        orig = row['smiles']
        try:
            std = standardize_smiles(orig)
            
            status = "OK"
            if std is None:
                failures += 1
                status = "FAIL"
            elif len(std) == 0:
                failures += 1
                status = "EMPTY"
            
            # Check if valid canonical output (basic check)
            if std:
                mol = Chem.MolFromSmiles(std)
                if not mol:
                    failures += 1
                    status = "INVALID_OUTPUT"

            # Truncate for display
            orig_disp = (orig[:32] + '...') if len(orig) > 32 else orig
            std_disp = (std[:32] + '...') if std and len(std) > 32 else (str(std) if std is not None else "None")
            
            print(f"{orig_disp:<35} | {std_disp:<35} | {status}")
            
        except Exception as e:
            print(f"{orig[:32]:<35} | ERROR: {str(e):<35} | CRASH")
            failures += 1

    print("-" * 80)
    failure_rate = (failures / total_samples) * 100
    print(f"\nSummary:")
    print(f"Total processed: {total_samples}")
    print(f"Failures: {failures}")
    print(f"Failure Rate: {failure_rate:.1f}%")

    if failure_rate > 5:
        print("\n[WARNING] Failure rate > 5%. Investigation required.")
    else:
        print("\n[SUCCESS] Verification passed (Failure rate <= 5%).")

if __name__ == "__main__":
    run_verification()
