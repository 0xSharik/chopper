
import pandas as pd
import numpy as np
import logging

# Configure logger
logger = logging.getLogger(__name__)

def load_and_prep_data(data_path):
    """
    Load features, validate, and separate X and y.
    
    Args:
        data_path (str): Path to feature CSV.
        
    Returns:
        tuple: (X, y, descriptor_cols, fingerprint_cols)
    """
    logger.info(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Validation
    if 'solubility' not in df.columns:
        raise ValueError("Missing 'solubility' (target) column")
        
    if df.isna().any().any():
        raise ValueError("NaN values found in dataset")
        
    # Separate types
    target_col = 'solubility'
    exclude_cols = ['smiles', 'standardized_smiles', target_col]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    fingerprint_cols = [c for c in feature_cols if c.startswith('fp_')]
    descriptor_cols = [c for c in feature_cols if c not in fingerprint_cols]
    
    # Validation
    if len(fingerprint_cols) != 2048:
        raise ValueError(f"Fingerprint column count mismatch: {len(fingerprint_cols)} != 2048")
        
    logger.info(f"Features loaded: {len(descriptor_cols)} descriptors, {len(fingerprint_cols)} fingerprint bits")
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y, descriptor_cols, fingerprint_cols

if __name__ == "__main__":
    # Test run
    import os
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, 'data/processed/esol_features.csv')
    
    try:
        X, y, desc_cols, fp_cols = load_and_prep_data(data_path)
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Descriptors: {len(desc_cols)}")
        print(f"Fingerprints: {len(fp_cols)}")
    except Exception as e:
        print(f"Error: {e}")
