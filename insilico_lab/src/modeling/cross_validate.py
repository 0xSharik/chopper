
import pandas as pd
import numpy as np
import os
import sys
import logging
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.modeling.data_prep import load_and_prep_data

def run_cross_validation(data_path, n_splits=5):
    logger.info(f"Starting {n_splits}-fold Cross-Validation")
    
    # 1. Load Data
    X, y, descriptor_cols, fingerprint_cols = load_and_prep_data(data_path)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    rmse_scores = []
    r2_scores = []
    
    fold = 1
    for train_index, val_index in kf.split(X):
        logger.info(f"Processing Fold {fold}/{n_splits}")
        
        # Split
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Scale Descriptors (Fit on Fold Train, Transform Fold Val)
        scaler = StandardScaler()
        
        X_train_desc = X_train[descriptor_cols]
        X_train_desc_scaled = scaler.fit_transform(X_train_desc)
        
        X_val_desc = X_val[descriptor_cols]
        X_val_desc_scaled = scaler.transform(X_val_desc)
        
        # Recombine
        X_train_fp = X_train[fingerprint_cols].values
        X_val_fp = X_val[fingerprint_cols].values
        
        X_train_fold = np.hstack([X_train_desc_scaled, X_train_fp])
        X_val_fold = np.hstack([X_val_desc_scaled, X_val_fp])
        
        # Train
        rf = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)
        rf.fit(X_train_fold, y_train)
        
        # Predict
        y_pred = rf.predict(X_val_fold)
        
        # Score
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        
        logger.info(f"Fold {fold} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        fold += 1
        
    # Summary
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    mean_r2 = np.mean(r2_scores)
    
    print("\n--- Cross-Validation Results ---")
    print(f"Mean RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")
    print(f"Mean R2:   {mean_r2:.4f}")
    
    if mean_rmse > 1.0:
        logger.warning("Mean RMSE > 1.0 in CV. Model accuracy is low.")
    if std_rmse > 0.2:
        logger.warning("High validation variance (unstable model).")
        
    return mean_rmse, std_rmse, mean_r2

if __name__ == "__main__":
    data_file = os.path.join(project_root, 'data/processed/esol_features.csv')
    run_cross_validation(data_file)
