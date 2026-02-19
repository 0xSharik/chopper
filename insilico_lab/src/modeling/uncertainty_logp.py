
import pandas as pd
import numpy as np
import os
import sys
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

def train_uncertainty_model(data_path, model_dir):
    logger.info("Starting LogP Random Forest training (for uncertainty)...")
    
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
    
    # 2. Split (80/20) - MUST match XGB split for fair comparison if we were comparing, 
    # but here we just need a good model.
    # ideally we use the same split, but random_state=42 should ensure it.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # 3. Scaling (Descriptors only)
    # We should load the scaler fitted by XGB if possible, or refit. 
    # Refitting on same data with same split is identical.
    # But for safety, let's load or refit. Refitting is easier here as self-contained.
    scaler = StandardScaler()
    X_train_desc = scaler.fit_transform(X_train[descriptor_cols])
    X_test_desc = scaler.transform(X_test[descriptor_cols])
    
    X_train_final = np.hstack([X_train_desc, X_train[fingerprint_cols].values])
    X_test_final = np.hstack([X_test_desc, X_test[fingerprint_cols].values])
    
    # 4. Train Random Forest
    # Config from ESOL: n_estimators=100 (or more for robustness)
    logger.info("Training RandomForestRegressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    )
    
    rf_model.fit(X_train_final, y_train)
    
    # 5. Evaluate
    y_pred_test = rf_model.predict(X_test_final)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print("\n--- LogP RF Performance (Uncertainty Provider) ---")
    print(f"Test RMSE:  {rmse_test:.4f}, R2: {r2_test:.4f}")
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'logp_rf.joblib')
    joblib.dump(rf_model, model_path)
    logger.info(f"RF model saved to {model_path}")
    
    return rmse_test

if __name__ == "__main__":
    data_file = os.path.join(project_root, 'data/processed/lipophilicity_features.csv')
    model_dir = os.path.join(project_root, 'models')
    
    train_uncertainty_model(data_file, model_dir)
