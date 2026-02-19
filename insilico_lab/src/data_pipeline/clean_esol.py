import pandas as pd
import numpy as np
import os
import sys
import logging

# Add src to path if running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir)) # adjusted for project root execution context if needed
# But usually better to assume correct python path or relative imports if package structure is sound
# Let's use relative imports assuming this is run as a module or with src in path
# If running as script, we might need to adjust sys.path
sys.path.append(os.path.join(current_dir, '../..'))

from src.data_pipeline.load_data import load_data
from src.data_pipeline.standardize import standardize_smiles
from src.utils.logging_config import setup_logging

# Setup logger
setup_logging()
logger = logging.getLogger(__name__)

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    initial_count = len(df)
    df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = initial_count - len(df_cleaned)
    
    return df_cleaned, removed_count

def clean_esol_data(input_path, output_path):
    """
    Clean the ESOL dataset.
    
    Pipeline:
    - Load dataset
    - Standardize SMILES
    - Remove invalid molecules
    - Remove duplicates
    - Remove solubility outliers
    - Save cleaned dataset
    """
    logger.info(f"Starting data cleaning for {input_path}")
    
    # 1. Load Data
    try:
        df = load_data(input_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    original_count = len(df)
    print(f"Original count: {original_count}")
    
    # 2. Standardize SMILES
    logger.info("Standardizing SMILES...")
    # Using apply with error handling inside standardize_smiles (returns None on failure)
    df['standardized_smiles'] = df['smiles'].apply(standardize_smiles)
    
    # 3. Remove Invalid Molecules
    df_valid = df.dropna(subset=['standardized_smiles'])
    invalid_removed = len(df) - len(df_valid)
    print(f"Invalid molecules removed: {invalid_removed}")
    
    if len(df_valid) == 0:
        logger.error("No valid molecules remaining after standardization.")
        return

    # 4. Remove Duplicates
    # Keep first occurrence
    df_no_dupes = df_valid.drop_duplicates(subset=['standardized_smiles'], keep='first')
    duplicates_removed = len(df_valid) - len(df_no_dupes)
    print(f"Duplicates removed: {duplicates_removed}")
    
    # 5. Remove Solubility Outliers
    df_cleaned, outliers_removed = remove_outliers_iqr(df_no_dupes, 'solubility')
    print(f"Outliers removed (IQR method): {outliers_removed}")
    
    final_count = len(df_cleaned)
    print(f"Final count: {final_count}")
    
    # 6. Save Cleaned Dataset
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save useful columns: original id, cleaned smiles, solubility
        # Just keeping all original columns plus standardized smiles for traceability, 
        # or maybe replacing smiles with standardized ones?
        # Requirement implies "Cleaned dataset". Usually best to keep standardized smiles as primary 'smiles' column or separate.
        # Let's save standardized_smiles and solubility.
        
        # Mapping back to requested output format? 
        # "Save cleaned dataset" -> usually implies saving the resulting dataframe.
        df_cleaned.to_csv(output_path, index=False)
        logger.info(f"Cleaned dataset saved to {output_path}")
        print(f"Cleaned dataset saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save cleaned data: {e}")

if __name__ == "__main__":
    # Define paths
    input_file = os.path.join(current_dir, '../../data/raw/esol.csv')
    output_file = os.path.join(current_dir, '../../data/processed/esol_cleaned.csv')
    
    clean_esol_data(input_file, output_file)
