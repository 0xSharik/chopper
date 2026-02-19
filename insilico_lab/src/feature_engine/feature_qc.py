
import pandas as pd
import numpy as np
import os
import json
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def qc_features(input_path, stats_output_path):
    logger.info(f"Checking features in {input_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"Feature file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    total_rows, total_cols = df.shape
    logger.info(f"Loaded {total_rows} rows, {total_cols} columns")

    # 1. Check feature count
    # fp_ columns
    fp_cols = [c for c in df.columns if c.startswith('fp_')]
    # descriptor columns: excludes metadata and fp_
    desc_cols = [c for c in df.columns if c not in fp_cols and c not in ['smiles', 'solubility', 'standardized_smiles']]
    
    print(f"Total feature columns: {total_cols}")
    print(f"Fingerprint columns: {len(fp_cols)}")
    print(f"Descriptor columns: {len(desc_cols)}")
    print(f"Descriptors: {desc_cols}")
    
    if len(fp_cols) != 2048:
        logger.error(f"Fingerprint column count mismatch: {len(fp_cols)} != 2048")
    
    # 2. Check NaNs
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logger.error(f"Found {nan_count} NaN values in dataset")
    else:
        print("[PASS] No NaN values found.")

    # 3. Check Infinite
    numeric_df = df.select_dtypes(include=[np.number])
    inf_count = np.isinf(numeric_df).sum().sum()
    if inf_count > 0:
         logger.error(f"Found {inf_count} infinite values")
    else:
        print("[PASS] No infinite values found.")

    # 4. Target stats
    mean_logS = df['solubility'].mean()
    std_logS = df['solubility'].std()
    
    print(f"Mean logS: {mean_logS:.4f}")
    print(f"Std logS: {std_logS:.4f}")

    # Save stats
    stats = {
        "total_rows": int(total_rows),
        "total_columns": int(total_cols),
        "fingerprint_columns": int(len(fp_cols)),
        "descriptor_columns": int(len(desc_cols)),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "mean_logS": float(mean_logS),
        "std_logS": float(std_logS)
    }
    
    os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)

    try:
        with open(stats_output_path, 'w') as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Feature stats saved to {stats_output_path}")
    except Exception as e:
        logger.error(f"Failed to save stats: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    input_file = os.path.join(project_root, 'data/processed/esol_features.csv')
    stats_file = os.path.join(project_root, 'data/metadata/esol_feature_stats.json')
    
    qc_features(input_file, stats_file)
