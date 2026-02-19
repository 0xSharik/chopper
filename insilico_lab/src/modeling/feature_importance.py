
import pandas as pd
import numpy as np
import os
import sys
import logging
import joblib

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.modeling.data_prep import load_and_prep_data

def export_feature_importance(data_path, model_path, output_path):
    logger.info("Exporting feature importance...")
    
    if not os.path.exists(model_path):
         logger.error(f"Model not found: {model_path}")
         return
         
    model = joblib.load(model_path)
    
    # Get feature names
    _, _, descriptor_cols, fingerprint_cols = load_and_prep_data(data_path)
    feature_names = descriptor_cols + fingerprint_cols
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        logger.error("Model does not have feature_importances_ attribute")
        return
        
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(20)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feat_imp.to_csv(output_path, index=False)
    logger.info(f"Top 20 features saved to {output_path}")
    print(feat_imp)

if __name__ == "__main__":
    data_file = os.path.join(project_root, 'data/processed/esol_features.csv')
    model_file = os.path.join(project_root, 'models/esol_rf.joblib')
    output_file = os.path.join(project_root, 'data/metadata/esol_top_features.csv')
    
    export_feature_importance(data_file, model_file, output_file)
