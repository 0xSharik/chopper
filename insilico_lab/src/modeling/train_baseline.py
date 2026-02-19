
import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
import sys
sys.path.append(project_root)

from src.modeling.data_prep import load_and_prep_data

def train_baseline_model(data_path, model_dir):
    logger.info("Starting baseline model training")
    
    # 1. Load Data
    X, y, descriptor_cols, fingerprint_cols = load_and_prep_data(data_path)
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 3. Scaling
    # Scale ONLY descriptors
    logger.info("Scaling descriptors...")
    scaler = StandardScaler()
    
    # Fit on/Transform Train descriptors
    X_train_desc = X_train[descriptor_cols]
    X_train_desc_scaled = scaler.fit_transform(X_train_desc)
    
    # Transform Test descriptors
    X_test_desc = X_test[descriptor_cols]
    X_test_desc_scaled = scaler.transform(X_test_desc)
    
    # Recombine with fingerprints (which are already 0/1, no scaling needed)
    # Convert scaled arrays back to DF to concatenate easily, or just use numpy concat
    # Using numpy for efficiency in model training
    X_train_fp = X_train[fingerprint_cols].values
    X_test_fp = X_test[fingerprint_cols].values
    
    X_train_final = np.hstack([X_train_desc_scaled, X_train_fp])
    X_test_final = np.hstack([X_test_desc_scaled, X_test_fp])
    
    # Save scaler
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, 'esol_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # 4. Model Training
    logger.info("Training RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)
    rf.fit(X_train_final, y_train)
    
    # Save model
    model_path = os.path.join(model_dir, 'esol_rf.joblib')
    joblib.dump(rf, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # 5. Evaluation
    logger.info("Evaluating model...")
    
    def evaluate(model, X, y_true, set_name="Test"):
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n{set_name} Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")
        
        return rmse, mae, r2, y_pred

    # Eval on Train
    rmse_train, _, _, _ = evaluate(rf, X_train_final, y_train, "Train")
    
    # Eval on Test
    rmse_test, _, _, y_pred_test = evaluate(rf, X_test_final, y_test, "Test")
    
    # Checks
    if rmse_test > 1.5:
        logger.warning("Weak model performance (RMSE > 1.5)")
    if rmse_test < 0.4:
         logger.warning("Suspiciously low RMSE (< 0.4)")
         
    return rf, scaler, X_test_final, y_test, y_pred_test

if __name__ == "__main__":
    data_file = os.path.join(project_root, 'data/processed/esol_features.csv')
    model_dir = os.path.join(project_root, 'models')
    
    train_baseline_model(data_file, model_dir)
