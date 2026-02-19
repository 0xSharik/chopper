
import pandas as pd
import numpy as np
import os
import sys
import logging
from tqdm import tqdm

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir # d:/chopper/project/insilico_lab
sys.path.append(project_root)

from src.data_pipeline.standardize import standardize_smiles
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def clean_caco2(input_path, output_path):
    logger.info(f"Cleaning Caco-2 data from {input_path}...")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    if 'smiles' not in df.columns or 'logPapp' not in df.columns:
        logger.error("Caco-2 missing columns. Need 'smiles' and 'logPapp'")
        return

    original_count = len(df)
    logger.info(f"Original rows: {original_count}")

    # Standardize
    tqdm.pandas(desc="Standardizing Caco-2 SMILES")
    df['standardized_smiles'] = df['smiles'].progress_apply(standardize_smiles)
    
    # Drop invalid
    df = df.dropna(subset=['standardized_smiles', 'logPapp'])
    valid_smiles_count = len(df)
    logger.info(f"Valid SMILES: {valid_smiles_count}")

    # Agg mean for duplicates
    df = df.groupby('standardized_smiles', as_index=False).agg({'logPapp': 'mean'})
    dedup_count = len(df)
    logger.info(f"Consolidated duplicates: {dedup_count}")

    # Outlier removal (range -10 to -2)
    df = df[(df['logPapp'] >= -10) & (df['logPapp'] <= -2)]
    range_filtered = len(df)
    logger.info(f"Range filtered (-10 to -2): {range_filtered}")

    # Remove IQR outliers
    Q1 = df['logPapp'].quantile(0.25)
    Q3 = df['logPapp'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['logPapp'] >= lower_bound) & (df['logPapp'] <= upper_bound)]
    
    final_count = len(df)
    logger.info(f"Final Cleaned Caco-2: {final_count}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def clean_bbb(input_path, output_path):
    logger.info(f"Cleaning BBB data from {input_path}...")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    # DeepChem BBBP has 'smiles' and 'p_np' (1/0)
    # Check columns
    if 'p_np' in df.columns:
        target_col = 'p_np'
    elif 'BBB' in df.columns: # Some versions
        target_col = 'BBB'
    else:
        # User said "p_np" is common
        # Let's inspect or assume index 0/1 if explicit names fail? No.
        logger.warning(f"Columns found: {df.columns.tolist()}. Assuming 'p_np' is target.")
        target_col = 'p_np'
    
    if target_col not in df.columns:
        logger.error("Target column not found in BBB data")
        return

    df = df.rename(columns={target_col: 'BBB'})
    target_col = 'BBB'

    original_count = len(df)
    logger.info(f"Original rows: {original_count}")

    tqdm.pandas(desc="Standardizing BBB SMILES")
    df['standardized_smiles'] = df['smiles'].progress_apply(standardize_smiles)

    df = df.dropna(subset=['standardized_smiles', 'BBB'])
    
    # Duplicates: Remove conflicting labels
    # If same SMILES has different labels, drop it.
    # Group by smiles, check nunique of BBB
    dup_groups = df.groupby('standardized_smiles')['BBB'].nunique()
    conflicts = dup_groups[dup_groups > 1].index
    logger.info(f"Removing {len(conflicts)} molecules with conflicting BBB labels")
    df = df[~df['standardized_smiles'].isin(conflicts)]
    
    # Dedup remaining (same label)
    df = df.drop_duplicates(subset=['standardized_smiles'])
    
    final_count = len(df)
    class_balance = df['BBB'].value_counts(normalize=True)
    logger.info(f"Final Cleaned BBB: {final_count}")
    logger.info(f"Class Balance: \n{class_balance}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_caco2(
        os.path.join(project_root, 'data/raw/caco2.csv'),
        os.path.join(project_root, 'data/processed/caco2_cleaned.csv')
    )
    clean_bbb(
        os.path.join(project_root, 'data/raw/bbbp.csv'),
        os.path.join(project_root, 'data/processed/bbbp_cleaned.csv')
    )
