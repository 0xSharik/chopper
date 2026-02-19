
import pandas as pd
import os
import sys
import logging
from tqdm import tqdm

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir # d:/chopper/project/insilico_lab
sys.path.append(project_root)

from src.feature_engine.descriptor_engine import generate_descriptors
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def build_features(input_path, output_path, target_col):
    logger.info(f"Building ADME features from {input_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    total_processed = len(df)
    
    features_list = []
    failed_count = 0

    for idx, row in tqdm(df.iterrows(), total=total_processed, desc=f"Generating features for {os.path.basename(input_path)}"):
        smiles = row['standardized_smiles']
        target = row[target_col]
        
        try:
            desc = generate_descriptors(smiles)
            if desc is None:
                failed_count += 1
                continue
            
            feature_row = {
                'smiles': smiles,
                target_col: target
            }
            feature_row.update(desc)
            features_list.append(feature_row)
            
        except Exception as e:
            logger.error(f"Failed to process {smiles}: {e}")
            failed_count += 1

    if not features_list:
        logger.error("No features generated")
        return

    final_df = pd.DataFrame(features_list)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved features to {output_path}")
    print(f"Total processed: {total_processed}")
    print(f"Failed: {failed_count}")
    print(f"Final Count: {len(final_df)}")

if __name__ == "__main__":
    # Caco-2
    build_features(
        os.path.join(project_root, 'data/processed/caco2_cleaned.csv'),
        os.path.join(project_root, 'data/processed/caco2_features.csv'),
        target_col='logPapp'
    )
    # BBB
    build_features(
        os.path.join(project_root, 'data/processed/bbbp_cleaned.csv'),
        os.path.join(project_root, 'data/processed/bbbp_features.csv'),
        target_col='BBB'
    )
