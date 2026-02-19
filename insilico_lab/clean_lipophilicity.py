
import pandas as pd
import os
import sys
import logging

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir # d:/chopper/project/insilico_lab
sys.path.append(project_root)

from src.data_pipeline.standardize import standardize_smiles
from src.data_pipeline.clean_esol import remove_outliers_iqr
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def clean_lipophilicity(input_path, output_path):
    logger.info("Starting Lipophilicity data cleaning pipeline...")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    initial_count = len(df)
    logger.info(f"Loaded {initial_count} rows")
    
    # Rename target column for consistency if needed
    # Target is 'exp' -> rename to 'logP'
    if 'exp' in df.columns:
        df = df.rename(columns={'exp': 'logP'})
    
    # 1. Standardize SMILES
    logger.info("Standardizing SMILES...")
    df['standardized_smiles'] = df['smiles'].apply(standardize_smiles)
    
    # Remove failures
    df_clean = df.dropna(subset=['standardized_smiles'])
    invalid_count = initial_count - len(df_clean)
    logger.info(f"Removed {invalid_count} invalid molecules")
    
    # 2. Remove Duplicates
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['standardized_smiles'])
    dedup_count = before_dedup - len(df_clean)
    logger.info(f"Removed {dedup_count} duplicates")
    
    # 3. Remove Outliers (Target range check + IQR)
    # Typical logP: -5 to +8 approx.
    # Let's use IQR method as before
    df_clean, outlier_count = remove_outliers_iqr(df_clean, 'logP')
    logger.info(f"Removed {outlier_count} outliers (IQR method)")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    final_count = len(df_clean)
    logger.info(f"Cleaning complete. Final count: {final_count}")
    print(f"Original: {initial_count}")
    print(f"Invalid: {invalid_count}")
    print(f"Duplicates: {dedup_count}")
    print(f"Outliers: {outlier_count}")
    print(f"Final: {final_count}")

if __name__ == "__main__":
    input_file = os.path.join(project_root, 'data/raw/lipophilicity.csv')
    output_file = os.path.join(project_root, 'data/processed/lipophilicity_cleaned.csv')
    
    clean_lipophilicity(input_file, output_file)
