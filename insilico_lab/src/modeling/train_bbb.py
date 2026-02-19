
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys
import logging
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def train_bbb_model(data_path, models_dir):
    logger.info("Starting BBB model training...")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Features & Target
    exclude_cols = ['smiles', 'BBB', 'standardized_smiles']
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('error')]
    
    X = df[feature_cols]
    y = df['BBB']
    
    # Handle infinite/NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    logger.info(f"Training with {X.shape[1]} features")

    # 2. Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'bbb_scaler.joblib'))
    
    # 4. Train XGBoost Classifier
    # Tuning for classification (imbalanced?)
    # Calculate scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    logger.info(f"Class ratio: {neg} Neg / {pos} Pos. Scale weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='auc'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=100
    )
    
    # 5. Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test ROC-AUC: {roc_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    logger.info(f"Classification Report:\n{report}")
    
    # 6. Save Model
    model_path = os.path.join(models_dir, 'bbb_model.json')
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

    # 7. Feature Importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(20)
    print("\nTop 20 Features:")
    print(feat_imp)
    
    # Save metrics
    metrics = {'accuracy': acc, 'roc_auc': roc_auc}
    with open(os.path.join(models_dir, 'bbb_metrics.json'), 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    train_bbb_model(
        os.path.join(project_root, 'data/processed/bbbp_features.csv'),
        os.path.join(project_root, 'models/bbb')
    )
