
import os
import sys
import logging
import pandas as pd
import requests

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def fetch_caco2_tdc(output_dir):
    """Fetches Caco-2 data from TDC (PyTDC) or direct download."""
    logger.info("Fetching Caco-2 (TDC Caco2_Wang)...")
    
    # Try PyTDC
    try:
        from tdc.single_pred import ADME
        data = ADME(name='Caco2_Wang')
        split = data.get_split()
        train, valid, test = split['train'], split['valid'], split['test']
        df = pd.concat([train, valid, test])
        # TDC standard columns: Drug_ID, Drug, Y
        # Rename 'Drug' to 'smiles', 'Y' to 'logPapp'
        if 'Drug' in df.columns and 'Y' in df.columns:
            df = df.rename(columns={'Drug': 'smiles', 'Y': 'logPapp'})
        
        output_path = os.path.join(output_dir, 'caco2.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Caco-2 (TDC) downloaded via PyTDC: {len(df)} rows")
        return True
    
    except ImportError:
        logger.warning("PyTDC not installed/import error. Using S3 direct download...")
    except Exception as e:
        logger.warning(f"PyTDC fetch failed: {e}. Using S3 direct download...")

    # Direct Download Fallback
    try:
        url = "https://tdc-data.s3.us-east-2.amazonaws.com/adme/Caco2_Wang.csv"
        df = pd.read_csv(url)
        
        # Standardize columns
        if 'Drug' in df.columns and 'Y' in df.columns:
            df = df.rename(columns={'Drug': 'smiles', 'Y': 'logPapp'})
        elif 'smiles' not in df.columns: # Heuristic
            # Check if columns look like ID, Smiles, Target
            # Often index 1 is smiles, 2 is target
            pass 
        
        output_path = os.path.join(output_dir, 'caco2.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Caco-2 (TDC) downloaded directly: {len(df)} rows")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Caco-2 from S3: {e}")
        return False

def fetch_bbb_deepchem(output_dir):
    """Fetches BBB data from DeepChem/MoleculeNet."""
    logger.info("Fetching BBB (DeepChem)...")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
    try:
        df = pd.read_csv(url)
        output_path = os.path.join(output_dir, 'bbbp.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"BBB downloaded: {len(df)} rows")
        return True
    except Exception as e:
        logger.error(f"Failed to download BBB: {e}")
        return False

def main():
    raw_dir = os.path.join(project_root, 'data/raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    fetch_caco2_tdc(raw_dir)
    fetch_bbb_deepchem(raw_dir)

if __name__ == "__main__":
    main()
