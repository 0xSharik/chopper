
import pandas as pd
import numpy as np
import os
import sys
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, 'src')) 
# Assuming script run from project root, but let's be safe and use relative inputs if needed or just assume src is importable

from src.data_pipeline.descriptor_engine import calculate_descriptors
from src.utils.logging_config import setup_logging

# Setup logger
setup_logging(log_file='model_training.log')
logger = logging.getLogger(__name__)

def train_esol_model(data_path, model_path):
    """
    Train a Random Forest Regressor on the ESOL dataset.
    """
    logger.info("Starting model training pipeline...")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {len(df)} samples")
    
    # 2. Generate Descriptors
    logger.info("Generating descriptors...")
    X = []
    y = []
    valid_indices = []
    
    # Use tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating descriptors"):
        smiles = row['standardized_smiles'] # Use standardized smiles
        target = row['solubility']
        
        desc = calculate_descriptors(smiles)
        
        if desc:
            # Flatten features: Physico-chemical + Fingerprint
            # Extract scalar features in order
            features = [
                desc['MW'], desc['TPSA'], desc['HBD'], desc['HBA'], 
                desc['RotBonds'], desc['AroRings'], desc['LogP']
            ]
            # Add fingerprint bits
            features.extend(desc['MorganFP'])
            
            X.append(features)
            y.append(target)
            valid_indices.append(idx)
        else:
            logger.warning(f"Failed to calculate descriptors for index {idx}: {smiles}")

    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"valid samples for training: {len(X)}")
    
    if len(X) == 0:
        logger.error("No valid data for training.")
        return

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 4. Train Model
    logger.info("Training RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # 5. Evaluate
    logger.info("Evaluating model...")
    y_pred = rf.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }
    
    print("\n--- Model Performance ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    logger.info(f"Metrics: {metrics}")
    
    # Validation checks against expected performance
    if rmse > 2.0:
        logger.warning(f"RMSE {rmse:.4f} is unusually high (> 2.0). Investigation needed.")
        print("[WARNING] RMSE > 2.0. Model performance is poor.")
    elif rmse < 0.5:
         # Too good to be true?
         pass # Actually 0.5 is great, typically 0.6-1.0
    
    # 6. Save Model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(rf, model_path)
    logger.info(f"Model saved to {model_path}")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in project root for execution
    data_file = os.path.join(base_dir, 'data/processed/esol_cleaned.csv')
    model_file = os.path.join(base_dir, 'models/esol_rf.joblib')
    
    train_esol_model(data_file, model_file)
