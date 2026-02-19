
import pandas as pd
import os
import sys
import logging
import numpy as np

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir # d:/chopper/project/insilico_lab
sys.path.append(project_root)

from src.data_pipeline.standardize import standardize_smiles
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def clean_pka(input_path, output_path):
    logger.info("Starting pKa data cleaning pipeline...")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    # Use low_memory=False to handle mixed types if any
    df = pd.read_csv(input_path, low_memory=False)
    initial_count = len(df)
    logger.info(f"Loaded {initial_count} rows")
    
    # 1. Rename target and standardize
    # Target is 'pka_value' -> rename to 'pKa'
    if 'pka_value' in df.columns:
        df = df.rename(columns={'pka_value': 'pKa'})
    else:
        logger.error("pka_value column not found")
        return
        
    # Ensure numerical
    df['pKa'] = pd.to_numeric(df['pKa'], errors='coerce')
    df = df.dropna(subset=['pKa'])
    
    # 2. Standardize SMILES
    logger.info("Standardizing SMILES...")
    if 'SMILES' in df.columns:
         df['smiles'] = df['SMILES']
    
    df['standardized_smiles'] = df['smiles'].apply(standardize_smiles)
    
    # Remove failures
    df_clean = df.dropna(subset=['standardized_smiles'])
    invalid_count = initial_count - len(df_clean) # Note: this includes non-numeric pKa drops too
    
    # 3. Remove Duplicates
    # For pKa, duplicates might have different values (experimental variation).
    # We should average them or take the median?
    # User instruction says: "Remove duplicates". 
    # Let's take the mean pKa for duplicates to be scientifically robust.
    before_dedup = len(df_clean)
    df_clean = df_clean.groupby('standardized_smiles', as_index=False)['pKa'].mean()
    dedup_count = before_dedup - len(df_clean)
    logger.info(f"Consolidated {dedup_count} duplicates (averaged pKa)")
    
    # 4. Remove Outliers (-5 to 20 range as per instruction)
    before_outlier = len(df_clean)
    df_clean = df_clean[(df_clean['pKa'] >= -5) & (df_clean['pKa'] <= 20)]
    outlier_count = before_outlier - len(df_clean)
    logger.info(f"Removed {outlier_count} outliers (range -5 to 20)")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    final_count = len(df_clean)
    logger.info(f"Cleaning complete. Final count: {final_count}")
    print(f"Original: {initial_count}")
    print(f"Final: {final_count}")

if __name__ == "__main__":
    input_file = os.path.join(project_root, 'data/raw/pka.csv')
    output_file = os.path.join(project_root, 'data/processed/pka_cleaned.csv')
    
    clean_pka(input_file, output_file)
