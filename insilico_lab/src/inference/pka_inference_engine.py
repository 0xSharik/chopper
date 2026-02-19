
import pandas as pd
import numpy as np
import os
import sys
import logging
import joblib
import xgboost as xgb
from rdkit import Chem

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.feature_engine.descriptor_engine import generate_descriptors
from src.modeling.uncertainty_module import UncertaintyEstimator
from src.modeling.applicability_domain import ApplicabilityDomain

class pKaInference:
    def __init__(self, model_dir=None, data_path=None):
        if model_dir is None:
            model_dir = os.path.join(project_root, 'models')
        if data_path is None:
            data_path = os.path.join(project_root, 'data/processed/pka_features.csv')
            
        self.xgb_path = os.path.join(model_dir, 'pka_xgb.joblib')
        self.rf_path = os.path.join(model_dir, 'pka_rf.joblib')
        self.scaler_path = os.path.join(model_dir, 'pka_scaler.joblib')
        self.data_path = data_path
        
        self._load_artifacts()
        self._fit_domain()

    def _load_artifacts(self):
        logger.info("Loading pKa models and scaler...")
        try:
            self.xgb_model = joblib.load(self.xgb_path)
            self.scaler = joblib.load(self.scaler_path)
            self.uncertainty_estimator = UncertaintyEstimator(self.rf_path)
            
            # Determine column order from data file if available
            # If training hasn't run yet, this will fail. Should be robust.
            # But inference implies training is done.
            if os.path.exists(self.data_path):
                df = pd.read_csv(self.data_path, nrows=5)
                exclude_cols = ['smiles', 'standardized_smiles', 'pKa']
                feature_cols = [c for c in df.columns if c not in exclude_cols]
                self.fingerprint_cols = [c for c in feature_cols if c.startswith('fp_')]
                self.descriptor_cols = [c for c in feature_cols if c not in self.fingerprint_cols]
            else:
                logger.warning("Data file not found, inference might fail if column order unknown.")
                
        except Exception as e:
            logger.error(f"Failed to load pKa artifacts: {e}")
            # Don't raise here if just initializing and training might happen later?
            # No, inference engine expects artifacts.
            raise

    def _fit_domain(self):
        logger.info("Fitting pKa applicability domain...")
        try:
            if os.path.exists(self.data_path):
                df = pd.read_csv(self.data_path)
                train_smiles = df['smiles'].tolist()
                self.ad = ApplicabilityDomain()
                self.ad.fit(train_smiles)
            else:
                logger.warning("Data file not found, skipping AD fit.")
                self.ad = None
        except Exception as e:
             logger.error(f"Failed to fit pKa applicability domain: {e}")
             raise

    def predict(self, smiles):
        """
        Predict pKa for a single SMILES.
        """
        try:
            if not smiles or not isinstance(smiles, str):
                return {'error': 'Invalid SMILES'}

            desc = generate_descriptors(smiles)
            if desc is None:
                return {'error': 'Descriptor generation failed'}
            
            # Extract features in correct order
            desc_values = []
            for col in self.descriptor_cols:
                if col not in desc:
                     # Check if it's one of the new descriptors and if missing (shouldn't be if generated correctly)
                     # But generate_descriptors was updated.
                     return {'error': f"Missing descriptor {col}"}
                desc_values.append(desc[col])
            
            fp_values = []
            for col in self.fingerprint_cols:
                 if col not in desc:
                     return {'error': "Missing fingerprint data"}
                 fp_values.append(desc[col])
            
            desc_array = np.array(desc_values).reshape(1, -1)
            desc_scaled = self.scaler.transform(desc_array)
            
            fp_array = np.array(fp_values).reshape(1, -1)
            X_final = np.hstack([desc_scaled, fp_array])
            
            xgb_pred = float(self.xgb_model.predict(X_final)[0])
            
            rf_res = self.uncertainty_estimator.predict_with_uncertainty(X_final)[0]
            rf_unc = rf_res['uncertainty_std']
            
            domain_status = "Unknown"
            if self.ad:
                ad_res = self.ad.predict([smiles])[0]
                domain_status = "In Domain" if ad_res['in_domain'] else "Out of Domain"
            
            return {
                "prediction": xgb_pred,
                "uncertainty": rf_unc,
                "domain_status": domain_status
            }

        except Exception as e:
            logger.error(f"pKa inference failed for {smiles}: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    engine = pKaInference()
    print(engine.predict("CC(=O)O")) # Acetic acid
