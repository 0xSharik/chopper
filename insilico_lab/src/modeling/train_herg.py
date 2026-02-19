
import pandas as pd
import numpy as np
import os
import sys
import logging
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def train_herg_model():
    """Train XGBoost classifier for hERG inhibition prediction."""
    
    # Load features
    features_path = os.path.join(project_root, 'data/processed/herg_features.csv')
    logger.info(f"Loading features from {features_path}")
    
    df = pd.read_csv(features_path)
    
    # Drop duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]
    
    logger.info(f"Loaded {len(df)} samples")
    
    # Separate features and target
    X = df.drop(['smiles', 'hERG'], axis=1)
    y = df['hERG']
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Class distribution:\n{y.value_counts(normalize=True)}")
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    logger.info("Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test ROC-AUC: {auc:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\nTop 10 Features:\n{feature_importance.head(10)}")
    
    # Save model and artifacts
    model_dir = os.path.join(project_root, 'models/herg')
    os.makedirs(model_dir, exist_ok=True)
    
    model.save_model(os.path.join(model_dir, 'herg_model.json'))
    joblib.dump(scaler, os.path.join(model_dir, 'herg_scaler.joblib'))
    
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(auc),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X.shape[1]
    }
    
    with open(os.path.join(model_dir, 'herg_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    feature_importance.to_csv(os.path.join(model_dir, 'herg_feature_importance.csv'), index=False)
    
    logger.info(f"Model saved to {model_dir}")
    
    return metrics

if __name__ == "__main__":
    train_herg_model()
