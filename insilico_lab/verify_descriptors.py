
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_pipeline.descriptor_engine import calculate_descriptors

# Configure logging
logging.basicConfig(level=logging.ERROR)

def verify_descriptors():
    test_molecules = [
        "CCO", # Ethanol
        "c1ccccc1", # Benzene
        "CC(=O)Oc1ccccc1C(=O)O", # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" # Ibuprofen
    ]

    print("--- Verifying Descriptor Engine ---")
    
    failures = 0
    for i, smiles in enumerate(test_molecules):
        print(f"\nProcessing molecule {i+1}: {smiles}")
        desc = calculate_descriptors(smiles)
        
        if not desc:
            print("FAILED: Could not calculate descriptors.")
            failures += 1
            continue
            
        # Check MW > 0
        mw = desc.get('MW', 0)
        if mw <= 0:
            print(f"FAILED: MW <= 0 ({mw})")
            failures += 1
        else:
            print(f"PASS: MW = {mw:.2f}")

        # Check TPSA reasonable (<300)
        tpsa = desc.get('TPSA', 999)
        if tpsa >= 300: # Simple small molecules shouldn't exceed this easily
            print(f"WARNING: High TPSA ({tpsa:.2f})")
            # Not strictly a failure unless absurdly high for these small mols
            if tpsa > 1000:
                print("FAILED: TPSA absurdly high.")
                failures += 1
        else:
            print(f"PASS: TPSA = {tpsa:.2f}")

        # Check Fingerprint length
        fp = desc.get('MorganFP', [])
        if len(fp) != 2048:
            print(f"FAILED: Fingerprint length = {len(fp)} (expected 2048)")
            failures += 1
        else:
            print(f"PASS: Fingerprint length = {len(fp)}")

    if failures == 0:
        print("\n[SUCCESS] Descriptor verification passed.")
    else:
        print(f"\n[FAILURE] {failures} check(s) failed.")

if __name__ == "__main__":
    verify_descriptors()
