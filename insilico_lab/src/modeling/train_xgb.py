
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

def train_xgb_model(data_path, model_dir):
    logger.info("Starting XGBoost training...")
    
    # 1. Load Data
    X, y, descriptor_cols, fingerprint_cols = load_and_prep_data(data_path)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # 3. Scaling (StandardScaler for descriptors)
    scaler = StandardScaler()
    X_train_desc = scaler.fit_transform(X_train[descriptor_cols])
    X_test_desc = scaler.transform(X_test[descriptor_cols])
    
    X_train_final = np.hstack([X_train_desc, X_train[fingerprint_cols].values])
    X_test_final = np.hstack([X_test_desc, X_test[fingerprint_cols].values])
    
    # 4. Train XGBoost
    # Config from requirements: n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8
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
    y_pred = xgb_model.predict(X_test_final)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- XGBoost Performance ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'esol_xgb.joblib') # Saving as joblib for consistency, though xgb has own format
    joblib.dump(xgb_model, model_path)
    logger.info(f"XGBoost model saved to {model_path}")
    
    return rmse, r2

if __name__ == "__main__":
    data_file = os.path.join(project_root, 'data/processed/esol_features.csv')
    model_dir = os.path.join(project_root, 'models')
    
    train_xgb_model(data_file, model_dir)
