
import pandas as pd
import os
import sys
import logging
from tqdm import tqdm

# Add src to path to allow importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.feature_engine.descriptor_engine import generate_descriptors
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def build_features(input_path, output_path):
    logger.info(f"Building features from {input_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    total_processed = len(df)
    logger.info(f"Loaded {total_processed} molecules")

    features_list = []
    failed_count = 0

    # Ensure required columns exist
    if 'standardized_smiles' not in df.columns:
         logger.error("Missing 'standardized_smiles' column")
         return
    if 'solubility' not in df.columns:
         logger.error("Missing 'solubility' column")
         return

    for idx, row in tqdm(df.iterrows(), total=total_processed, desc="Generating features"):
        smiles = row['standardized_smiles']
        target = row['solubility']
        
        try:
            desc = generate_descriptors(smiles)
            if desc is None:
                failed_count += 1
                continue
            
            # Combine identifying info and target with descriptors
            # Keep standardized_smiles and solubility
            feature_row = {
                'smiles': smiles,
                'solubility': target
            }
            feature_row.update(desc)
            features_list.append(feature_row)
            
        except Exception as e:
            logger.error(f"Failed to process {smiles}: {e}")
            failed_count += 1

    if not features_list:
        logger.error("No features generated")
        return

    try:
        final_df = pd.DataFrame(features_list)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        final_df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")
        
        print(f"Total processed: {total_processed}")
        print(f"Failed molecules: {failed_count}")
        print(f"Final dataset size: {len(final_df)}")
        
    except Exception as e:
        logger.error(f"Error saving features: {e}")

if __name__ == "__main__":
    input_file = os.path.join(project_root, 'data/processed/esol_cleaned.csv')
    output_file = os.path.join(project_root, 'data/processed/esol_features.csv')
    
    build_features(input_file, output_file)
