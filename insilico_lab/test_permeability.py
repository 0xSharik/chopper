
import sys
import os
import logging

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.inference.property_engine import PropertyEngine

def test_permeability():
    print("Initializing Property Engine (with Permeability)...")
    engine = PropertyEngine()
    
    # Test cases:
    # 1. Diazepam (CNS Active, High Perm, BBB+)
    # 2. Penicillin G (Low CNS, BBB-)
    # 3. Dopamine (Does not cross BBB efficiently)
    # 4. Glucose (Requires transporter, passive perm low, but might be tricky for simple models)
    
    test_cases = [
        ("Diazepam (BBB+)", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
        ("Penicillin G (BBB-)", "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"),
        ("Dopamine (BBB-)", "NCCc1ccc(O)c(O)c1"),
        ("Ethanol (BBB+)", "CCO"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ]
    
    print("\n--- Running Permeability Predictions ---\n")
    
    for name, smiles in test_cases:
        print(f"Testing: {name}")
        print(f"SMILES: {smiles}")
        
        try:
            res = engine.predict_properties(smiles)
            perm = res.get('Pe', {})
            
            # Caco-2
            caco2 = perm.get('Caco2', {})
            print(f"  [Caco-2] Pred: {caco2.get('prediction', 'N/A')} {caco2.get('unit', '')} (Status: {caco2.get('status')})")
            if 'error' in caco2:
                print(f"    Error: {caco2['error']}")

            # BBB
            bbb = perm.get('BBB', {})
            print(f"  [BBB]    Pred: {bbb.get('prediction', 'N/A')} (Prob: {bbb.get('probability', 'N/A'):.4f}) (Status: {bbb.get('status')})")
            if 'error' in bbb:
                print(f"    Error: {bbb['error']}")
                
        except Exception as e:
            print(f"  CRITICAL FAILURE: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_permeability()
