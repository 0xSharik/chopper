
import pandas as pd
import numpy as np
import os
import sys
import logging

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.inference.esol_inference_engine import ESOLInference

def run_batch_test(n_samples=50):
    print(f"--- STAGE 2.6: Batch Inference Test ({n_samples} samples) ---")
    
    project_root = current_dir # d:/chopper/project/insilico_lab
    data_path = os.path.join(project_root, 'data/processed/esol_features.csv')
    print(f"DEBUG: Current Dir: {current_dir}")
    print(f"DEBUG: Project Root: {project_root}")
    print(f"DEBUG: Data Path: {data_path}")
    if not os.path.exists(data_path):
        print("Data file not found.")
        return

    try:
        engine = ESOLInference()
    except Exception as e:
        print(f"FAILED TO INIT ENGINE: {e}")
        return
        
    df = pd.read_csv(data_path)
    
    # Stratified sampling would be better, but random is okay for quick check
    sample_df = df.sample(n=n_samples, random_state=42)
    
    results = []
    
    for idx, row in sample_df.iterrows():
        smiles = row['smiles']
        true_logS = row['solubility']
        
        res = engine.predict(smiles)
        
        if 'error' in res:
            print(f"Error for {smiles}: {res['error']}")
            continue
            
        xgb_pred = res['xgb_prediction']
        rf_unc = res['rf_uncertainty']
        
        error = abs(xgb_pred - true_logS)
        
        results.append({
            'smiles': smiles,
            'true_logS': true_logS,
            'xgb_pred': xgb_pred,
            'uncertainty': rf_unc,
            'abs_error': error,
            'in_domain': res['domain_status'] == 'In Domain'
        })
        
    if not results:
        print("No valid results.")
        return
        
    results_df = pd.DataFrame(results)
    
    mae = results_df['abs_error'].mean()
    mean_unc = results_df['uncertainty'].mean()
    in_domain_pct = results_df['in_domain'].mean() * 100
    
    # Correlation between error and uncertainty
    corr = results_df['abs_error'].corr(results_df['uncertainty'])
    
    print("\n--- Batch Results ---")
    print(f"MAE (XGB): {mae:.4f}")
    print(f"Mean Uncertainty: {mean_unc:.4f}")
    print(f"In Domain: {in_domain_pct:.1f}%")
    print(f"Error-Uncertainty Correlation: {corr:.4f}")
    
    print("\n--- Examples (Top 5 Highest Error) ---")
    print(results_df.sort_values('abs_error', ascending=False).head(5)[['smiles', 'true_logS', 'xgb_pred', 'abs_error', 'uncertainty']])

if __name__ == "__main__":
    run_batch_test()
