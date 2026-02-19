
import logging
import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem

from src.feature_engine.descriptor_engine import generate_descriptors

# Configure logger
logger = logging.getLogger(__name__)

class PermeabilityEngine:
    def __init__(self, model_dir):
        """
        Initialize the Permeability Engine.
        
        Args:
            model_dir (str): Path to the directory containing models and scalers.
        """
        self.model_dir = model_dir
        self.caco2_model = None
        self.caco2_scaler = None
        self.bbb_model = None
        self.bbb_scaler = None
        
        self.load_models()

    def load_models(self):
        """Load Caco-2 and BBB models and scalers."""
        try:
            # Caco-2 (Provisional/Synthetic)
            caco2_model_path = os.path.join(self.model_dir, 'caco2', 'caco2_model.json')
            caco2_scaler_path = os.path.join(self.model_dir, 'caco2', 'caco2_scaler.joblib')
            
            if os.path.exists(caco2_model_path) and os.path.exists(caco2_scaler_path):
                self.caco2_model = xgb.XGBRegressor()
                self.caco2_model.load_model(caco2_model_path)
                self.caco2_scaler = joblib.load(caco2_scaler_path)
                logger.info("Caco-2 model loaded.")
            else:
                logger.warning("Caco-2 model or scaler not found.")

            # BBB
            bbb_model_path = os.path.join(self.model_dir, 'bbb', 'bbb_model.json')
            bbb_scaler_path = os.path.join(self.model_dir, 'bbb', 'bbb_scaler.joblib')
            
            if os.path.exists(bbb_model_path) and os.path.exists(bbb_scaler_path):
                self.bbb_model = xgb.XGBClassifier()
                self.bbb_model.load_model(bbb_model_path)
                self.bbb_scaler = joblib.load(bbb_scaler_path)
                logger.info("BBB model loaded.")
            else:
                logger.warning("BBB model or scaler not found.")
                
        except Exception as e:
            logger.error(f"Error loading permeability models: {e}")

    def predict_permeability(self, smiles):
        """
        Predict Caco-2 permeability (logPapp) and BBB penetration.
        
        Args:
            smiles (str): SMILES string.
            
        Returns:
            dict: Predictions and confidence/uncertainty (if available).
        """
        result = {
            'Caco2': {'prediction': None, 'unit': 'logPapp (cm/s)', 'status': 'Error'},
            'BBB': {'prediction': None, 'probability': None, 'class': None, 'status': 'Error'}
        }
        
        try:
            # Generate Features
            features = generate_descriptors(smiles)
            if features is None:
                logger.warning(f"Feature generation failed for {smiles}")
                return result

            # Convert to DataFrame (1 row) to match training format
            # Ensure column order matches scaler (usually handled by pandas if names match, but here features is a dict)
            # We need to ensure the columns are in the same order/set as training.
            # Ideally we should save the feature names during training.
            # For now, we assume dict keys -> DataFrame columns works if we re-index or if the scaler accepts it.
            # Scaler usually expects numpy array. We need to align features.
            
            # Use a robust way: sorted keys or saved feature list.
            # Let's rely on the fact that `generate_descriptors` is deterministic dictionary.
            # However, XGBoost is sensitive to feature order if using DMatrix or arrays.
            # We should probably align with the scaler/model expectations.
            
            # Since we didn't save feature list explicitly in `train_*.py`, let's try to infer or re-use `generate_descriptors` output structure.
            # Training script loaded CSV. CSV column order comes from `build_adme_features.py` which puts dict keys into columns.
            # So `pd.DataFrame([features])` should have columns in roughly the same order? No, dict order is insertion order in py3.7+.
            # It should be consistent if generation code hasn't changed.
            
            df_feat = pd.DataFrame([features])
            
            # Align columns if possible (Training handles NaNs with 0, so we should too)
            df_feat = df_feat.fillna(0)
            
            # Caco-2 Prediction
            if self.caco2_model and self.caco2_scaler:
                try:
                    # Align with scaler needs. Scaler was fitted on DF column order of training set.
                    # We might have discrepancies if training dropped columns.
                    # Training dropped 'smiles', 'logPapp', 'standardized_smiles'.
                    # Features dict doesn't have these.
                    # So keys should match.
                    # Reorder to match scaler? Scaler stores `feature_names_in_` in sklearn v1.0+
                    
                    if hasattr(self.caco2_scaler, 'feature_names_in_'):
                        df_caco2 = df_feat[self.caco2_scaler.feature_names_in_]
                    else:
                        df_caco2 = df_feat # Fallback, risky if order changed
                        
                    X_caco2 = self.caco2_scaler.transform(df_caco2)
                    pred_logPapp = self.caco2_model.predict(X_caco2)[0]
                    
                    result['Caco2'] = {
                        'prediction': float(pred_logPapp),
                        'unit': 'logPapp (cm/s)',
                        'status': 'Success'
                    }
                except Exception as e:
                    logger.error(f"Caco-2 inference error: {e}")
                    result['Caco2']['error'] = str(e)

            # BBB Prediction
            if self.bbb_model and self.bbb_scaler:
                try:
                    if hasattr(self.bbb_scaler, 'feature_names_in_'):
                        df_bbb = df_feat[self.bbb_scaler.feature_names_in_]
                    else:
                        df_bbb = df_feat
                        
                    X_bbb = self.bbb_scaler.transform(df_bbb)
                    pred_prob = self.bbb_model.predict_proba(X_bbb)[0][1]
                    pred_class = int(pred_prob > 0.5) # Default threshold
                    
                    result['BBB'] = {
                        'prediction': 'Penetrant' if pred_class == 1 else 'Non-Penetrant',
                        'probability': float(pred_prob),
                        'class': pred_class,
                        'status': 'Success'
                    }
                except Exception as e:
                    logger.error(f"BBB inference error: {e}")
                    result['BBB']['error'] = str(e)
                    
        except Exception as e:
            logger.error(f"Prediction error for {smiles}: {e}")
            
        return result

if __name__ == "__main__":
    # Test
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    engine = PermeabilityEngine(os.path.join(project_root, 'models'))
    
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O" # Aspirin
    print(f"Testing Aspirin: {test_smiles}")
    res = engine.predict_permeability(test_smiles)
    print(res)
