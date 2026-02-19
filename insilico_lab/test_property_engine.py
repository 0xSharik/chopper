
import os
import sys

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # d:/chopper/project/insilico_lab
sys.path.append(project_root)

from src.inference.property_engine import PropertyEngine

def run_test():
    print("--- STAGE 3: Unified Property Engine Test ---")
    
    try:
        engine = PropertyEngine()
    except Exception as e:
        print(f"FAILED TO INIT ENGINE: {e}")
        return

    test_cases = [
        ("Ethanol (High Sol, Low logP)", "CCO"),
        ("Benzene (Low Sol, Mod logP)", "c1ccccc1"),
        ("Acetic Acid (Acidic, pKa ~4.75)", "CC(=O)O"),
        ("Methylamine (Basic, pKa ~10.6)", "CN"),
        ("Long Alkane (Very Low Sol, High logP)", "CCCCCCCCCCCCCCCCCCCCCCCCCCCC"),
        ("Invalid SMILES", "ABCXYZ")
    ]
    
    for name, smiles in test_cases:
        print(f"\nTesting: {name}")
        print(f"SMILES: {smiles}")
        
        try:
            res = engine.predict_properties(smiles)
            
            print("  [ESOL / Solubility]")
            if 'error' in res['logS']:
                print(f"    Error: {res['logS']['error']}")
            else:
                print(f"    Pred: {res['logS']['prediction']:.4f}")
                print(f"    Uncert: {res['logS']['uncertainty']:.4f}")
                print(f"    Domain: {res['logS']['domain_status']}")

            print("  [Lipophilicity / logP]")
            if 'error' in res['logP']:
                print(f"    Error: {res['logP']['error']}")
            else:
                print(f"    Pred: {res['logP']['prediction']:.4f}")
                print(f"    Uncert: {res['logP']['uncertainty']:.4f}")
                print(f"    Domain: {res['logP']['domain_status']}")

            print("  [Ionization / pKa]")
            if 'error' in res['pKa']:
                print(f"    Error: {res['pKa']['error']}")
            else:
                print(f"    Pred: {res['pKa']['prediction']:.4f}")
                print(f"    Uncert: {res['pKa']['uncertainty']:.4f}")
                print(f"    Domain: {res['pKa']['domain_status']}")
                
            print(f"  [Derived logD pH 7.4]: {res.get('logD_ph7.4', 'N/A')}")
                
        except Exception as e:
            print(f"  CRITICAL FAILURE: {e}")

if __name__ == "__main__":
    run_test()
