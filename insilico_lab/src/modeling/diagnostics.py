
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
import joblib

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.modeling.data_prep import load_and_prep_data
from src.modeling.train_baseline import train_baseline_model # Re-use training flow to get predictions if model not loaded

def run_diagnostics(data_path, model_path, scaler_path, output_dir):
    logger.info("Running model diagnostics")
    
    # Needs to replicate preprocessing to get comparable test set predictions
    # Or simply reload the trained model and run prediction on a fresh test split?
    # Ideally, diagnostics should run on the held-out test set used during training.
    # To ensure identical split, we must use same random_state.
    
    from src.modeling.train_baseline import train_baseline_model
    
    # We can just call train_baseline which returns predictions on test set
    # This avoids duplication of split/scaling logic
    # But it retrains the model. If model is already trained and saved, we should load it.
    # However, to get the EXACT X_test scaled correctly, we need the exact split and scaler.
    
    # Let's re-run the training pipeline to get the exact X_test and y_test and y_pred
    # It's fast for this dataset size.
    
    rf_model, scaler, X_test, y_test, y_pred = train_baseline_model(data_path, os.path.dirname(model_path))
    
    # 1. Predicted vs Actual
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Log Solubility')
    plt.ylabel('Predicted Log Solubility')
    plt.title('Predicted vs Actual (Test Set)')
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'esol_predictions.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Prediction plot saved to {plot_path}")
    
    # 2. Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual (Actual - Predicted)')
    plt.title('Residual Distribution')
    plt.axvline(0, color='r', linestyle='--')
    
    plot_path = os.path.join(output_dir, 'esol_residuals.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Residual plot saved to {plot_path}")
    
    # 3. Feature Importance
    # Get importance from model
    importances = rf_model.feature_importances_
    
    # Get feature names
    # Need to reconstruct them: descriptors + fp_0...fp_2047
    _, _, descriptor_cols, fingerprint_cols = load_and_prep_data(data_path)
    feature_names = descriptor_cols + fingerprint_cols
    
    # Create DataFrame
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(20)
    
    print("\n--- Top 20 Feature Importances ---")
    print(feat_imp)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp)
    plt.title('Top 20 Feature Importances')
    
    plot_path = os.path.join(output_dir, 'esol_feature_importance.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Feature importance plot saved to {plot_path}")

if __name__ == "__main__":
    data_file = os.path.join(project_root, 'data/processed/esol_features.csv')
    model_file = os.path.join(project_root, 'models/esol_rf.joblib')
    scaler_file = os.path.join(project_root, 'models/esol_scaler.joblib')
    metadata_dir = os.path.join(project_root, 'data/metadata')
    
    run_diagnostics(data_file, model_file, scaler_file, metadata_dir)
