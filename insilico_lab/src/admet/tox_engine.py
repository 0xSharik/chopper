
import os
import sys
import logging
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.feature_engine.descriptor_engine import generate_descriptors
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class ToxicityEngine:
    """Predicts toxicity risk using ClinTox model."""
    
    def __init__(self):
        self.clintox_model = None
        self.clintox_scaler = None
        self._load_models()
    
    def _load_models(self):
        """Load trained toxicity models."""
        try:
            # Load ClinTox model
            clintox_dir = os.path.join(project_root, 'models/clintox')
            self.clintox_model = xgb.XGBClassifier()
            self.clintox_model.load_model(os.path.join(clintox_dir, 'clintox_model.json'))
            self.clintox_scaler = joblib.load(os.path.join(clintox_dir, 'clintox_scaler.joblib'))
            logger.info("ClinTox model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ClinTox model: {e}")
            raise
    
    def predict(self, smiles):
        """
        Predict toxicity for a given SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            dict: Toxicity predictions
        """
        try:
            # Generate features
            features = generate_descriptors(smiles)
            if features is None:
                return {
                    'clintox': {'probability': None, 'class': 'Unknown', 'risk_level': 'Unknown'}
                }
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Drop duplicate columns
            feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]
            
            # ClinTox prediction
            clintox_result = self._predict_clintox(feature_df)
            
            return {
                'clintox': clintox_result
            }
            
        except Exception as e:
            logger.error(f"Toxicity prediction failed: {e}")
            return {
                'clintox': {'probability': None, 'class': 'Error', 'risk_level': 'Unknown'}
            }
    
    def _predict_clintox(self, feature_df):
        """Predict clinical toxicity."""
        try:
            # Convert to numpy to bypass sklearn feature name validation
            # The scaler expects the same feature order as training
            if hasattr(self.clintox_scaler, 'feature_names_in_'):
                # Reorder columns to match training
                feature_df = feature_df[self.clintox_scaler.feature_names_in_]
            
            # Scale features - convert to numpy to avoid feature name warnings
            features_scaled = self.clintox_scaler.transform(feature_df.values)
            
            # Predict
            prob = self.clintox_model.predict_proba(features_scaled)[0, 1]
            pred_class = int(self.clintox_model.predict(features_scaled)[0])
            
            # Risk level
            if prob < 0.3:
                risk_level = "Low"
            elif prob < 0.7:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            
            return {
                'probability': float(prob),
                'class': pred_class,
                'risk_level': risk_level
            }
        except Exception as e:
            logger.error(f"ClinTox prediction failed: {e}")
            return {'probability': None, 'class': 'Error', 'risk_level': 'Unknown'}

if __name__ == "__main__":
    # Test
    engine = ToxicityEngine()
    
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    ]
    
    for smi in test_smiles:
        result = engine.predict(smi)
        print(f"\nSMILES: {smi}")
        print(f"ClinTox: {result['clintox']}")
