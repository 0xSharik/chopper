
import os
import sys
import logging

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # d:/chopper/project/insilico_lab
sys.path.append(project_root)

from src.inference.esol_inference_engine import ESOLInference

def run_manual_test():
    print("--- STAGE 2.6: Manual Inference Test ---")
    
    try:
        engine = ESOLInference()
    except Exception as e:
        print(f"FAILED TO INIT ENGINE: {e}")
        return

    test_cases = [
        ("Ethanol (High Sol)", "CCO"),
        ("Benzene (Low Sol)", "c1ccccc1"),
        ("Long Alkane (Very Low Sol)", "CCCCCCCCCCCCCCCCCCCCCCCCCCCC"),
        ("Invalid SMILES", "ABCXYZ"),
        ("Naphthalene", "c1ccc2ccccc2c1")
    ]
    
    for name, smiles in test_cases:
        print(f"\nTesting: {name}")
        print(f"SMILES: {smiles}")
        
        res = engine.predict(smiles)
        
        if 'error' in res:
            print(f"RESULT: Error - {res['error']}")
        else:
            print(f"XGB Prediction: {res['xgb_prediction']:.4f}")
            print(f"RF Prediction:  {res['rf_prediction']:.4f}")
            print(f"RF Uncertainty: {res['rf_uncertainty']:.4f}")
            print(f"Max Similarity: {res['max_similarity']:.4f}")
            print(f"Domain Status:  {res['domain_status']}")

if __name__ == "__main__":
    run_manual_test()
