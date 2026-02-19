
import pandas as pd
import os
import sys
import logging

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

from src.data_pipeline.standardize import standardize_smiles
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def clean_tox_data():
    raw_dir = os.path.join(project_root, 'data/raw')
    proc_dir = os.path.join(project_root, 'data/processed')
    os.makedirs(proc_dir, exist_ok=True)
    
    # 1. Clean ClinTox
    logger.info("Cleaning ClinTox...")
    try:
        df = pd.read_csv(os.path.join(raw_dir, 'clintox_raw.csv'))
        # Columns: 'smiles', 'FDA_APPROVED', 'CT_TOX_FDA_APPROVED_FAILED'
        # We want 'CT_TOX_FDA_APPROVED_FAILED' (Clinical Toxicity during approval)
        # 1 = Toxicity, 0 = No Toxicity (presumably)
        
        # Check for CT_TOX or CT_TOX_FDA_APPROVED_FAILED
        if 'CT_TOX' in df.columns:
            target = 'CT_TOX'
        elif 'CT_TOX_FDA_APPROVED_FAILED' in df.columns:
            target = 'CT_TOX_FDA_APPROVED_FAILED'
        else:
            logger.error(f"ClinTox target column not found. Available: {df.columns.tolist()}")
            return
        
        df_clean = df[['smiles', target]].copy()
        df_clean = df_clean.dropna(subset=['smiles', target])
        
        # Standardize
        df_clean['standardized_smiles'] = df_clean['smiles'].apply(standardize_smiles)
        df_clean = df_clean.dropna(subset=['standardized_smiles'])
        
        # Label
        df_clean.rename(columns={target: 'ClinTox'}, inplace=True)
        
        # Deduplicate (If conflict, take max risk (1))
        df_clean = df_clean.sort_values('ClinTox', ascending=False).drop_duplicates('standardized_smiles', keep='first')
        
        df_clean.to_csv(os.path.join(proc_dir, 'clintox_cleaned.csv'), index=False)
        logger.info(f"ClinTox cleaned: {len(df_clean)} rows")

    except Exception as e:
        logger.error(f"ClinTox cleaning failed: {e}")

    # 2. Clean hERG
    logger.info("Cleaning hERG...")
    try:
        df = pd.read_csv(os.path.join(raw_dir, 'herg_raw.csv'))
        # If synthetic (from BBB), it has 'smiles' and 'hERG'
        
        if 'hERG' not in df.columns:
            logger.error("hERG column not found")
        else:
            df_clean = df[['smiles', 'hERG']].copy()
            df_clean = df_clean.dropna()
            
            df_clean['standardized_smiles'] = df_clean['smiles'].apply(standardize_smiles)
            df_clean = df_clean.dropna(subset=['standardized_smiles'])
            
            # Deduplicate
            df_clean = df_clean.drop_duplicates('standardized_smiles')
            
            df_clean.to_csv(os.path.join(proc_dir, 'herg_cleaned.csv'), index=False)
            logger.info(f"hERG cleaned: {len(df_clean)} rows")
            
    except Exception as e:
        logger.error(f"hERG cleaning failed: {e}")

if __name__ == "__main__":
    clean_tox_data()
