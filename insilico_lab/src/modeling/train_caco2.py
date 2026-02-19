
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys
import logging
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def train_caco2_model(data_path, models_dir):
    logger.info("Starting Caco-2 model training...")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Features & Target
    # Exclude non-feature columns
    exclude_cols = ['smiles', 'logPapp', 'standardized_smiles']
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('error')]
    
    X = df[feature_cols]
    y = df['logPapp']
    
    # Handle infinite/NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    logger.info(f"Training with {X.shape[1]} features")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'caco2_scaler.joblib'))
    
    # 4. Train XGBoost
    # Tuning for regression
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=100
    )
    
    # 5. Evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test R2: {r2:.4f}")
    
    # 6. Save Model
    model_path = os.path.join(models_dir, 'caco2_model.json')
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

    # 7. Feature Importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(20)
    print("\nTop 20 Features:")
    print(feat_imp)
    
    # Save metrics
    metrics = {'rmse': rmse, 'r2': r2}
    with open(os.path.join(models_dir, 'caco2_metrics.json'), 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    train_caco2_model(
        os.path.join(project_root, 'data/processed/caco2_features.csv'),
        os.path.join(project_root, 'models/caco2')
    )
