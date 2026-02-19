
import pandas as pd
import numpy as np
import os
import sys
import logging
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.modeling.data_prep import load_and_prep_data

def train_logp_model(data_path, model_dir):
    logger.info("Starting LogP XGBoost training...")
    
    # 1. Load Data
    # Note: load_and_prep_data looks for 'solubility' column hardcoded? 
    # Let's check src/modeling/data_prep.py
    # It constructs target from 'solubility'.
    # I should update data_prep to handle 'logP' or just split manually here since it's a new property.
    # To keep data_prep generic, I should probably modify it or write a custom loader here.
    # Given the task instruction "Reuse intelligently", let's make a local loader that is similar.
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows")
    
    target_col = 'logP'
    if target_col not in df.columns:
         logger.error(f"Missing target column '{target_col}'")
         return
         
    exclude_cols = ['smiles', 'standardized_smiles', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    fingerprint_cols = [c for c in feature_cols if c.startswith('fp_')]
    descriptor_cols = [c for c in feature_cols if c not in fingerprint_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 2. Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # 3. Scaling (Descriptors only)
    scaler = StandardScaler()
    X_train_desc = scaler.fit_transform(X_train[descriptor_cols])
    X_test_desc = scaler.transform(X_test[descriptor_cols])
    
    X_train_final = np.hstack([X_train_desc, X_train[fingerprint_cols].values])
    X_test_final = np.hstack([X_test_desc, X_test[fingerprint_cols].values])
    
    # Save scaler
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, 'logp_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # 4. Train XGBoost
    # Config: n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8
    logger.info("Training XGBRegressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train_final, y_train)
    
    # 5. Evaluate
    # Train metrics
    y_pred_train = xgb_model.predict(X_train_final)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)

    # Test metrics
    y_pred_test = xgb_model.predict(X_test_final)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print("\n--- LogP XGBoost Performance ---")
    print(f"Train RMSE: {rmse_train:.4f}, R2: {r2_train:.4f}")
    print(f"Test RMSE:  {rmse_test:.4f}, R2: {r2_test:.4f}")
    
    # Save model
    model_path = os.path.join(model_dir, 'logp_xgb.joblib')
    joblib.dump(xgb_model, model_path)
    logger.info(f"XGBoost model saved to {model_path}")
    
    return rmse_test, r2_test

if __name__ == "__main__":
    data_file = os.path.join(project_root, 'data/processed/lipophilicity_features.csv')
    model_dir = os.path.join(project_root, 'models')
    
    train_logp_model(data_file, model_dir)
