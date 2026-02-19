
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
from src.modeling.data_prep import load_and_prep_data
from src.modeling.uncertainty_module import UncertaintyEstimator
from src.modeling.applicability_domain import ApplicabilityDomain

class LogPInference:
    def __init__(self, model_dir=None, data_path=None):
        if model_dir is None:
            model_dir = os.path.join(project_root, 'models')
        if data_path is None:
            data_path = os.path.join(project_root, 'data/processed/lipophilicity_features.csv')
            
        self.xgb_path = os.path.join(model_dir, 'logp_xgb.joblib')
        self.rf_path = os.path.join(model_dir, 'logp_rf.joblib')
        self.scaler_path = os.path.join(model_dir, 'logp_scaler.joblib')
        self.data_path = data_path
        
        self._load_artifacts()
        self._fit_domain()

    def _load_artifacts(self):
        logger.info("Loading LogP models and scaler...")
        try:
            self.xgb_model = joblib.load(self.xgb_path)
            self.scaler = joblib.load(self.scaler_path)
            
            self.uncertainty_estimator = UncertaintyEstimator(self.rf_path)
            
            # Determine column order from data file
            df = pd.read_csv(self.data_path, nrows=5)
            # Remove target and smiles
            exclude_cols = ['smiles', 'standardized_smiles', 'logP'] # Check target name in file!
            # In clean_lipophilicity we renamed 'exp' to 'logP'.
            
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            self.fingerprint_cols = [c for c in feature_cols if c.startswith('fp_')]
            self.descriptor_cols = [c for c in feature_cols if c not in self.fingerprint_cols]
            
        except Exception as e:
            logger.error(f"Failed to load LogP artifacts: {e}")
            raise

    def _fit_domain(self):
        logger.info("Fitting LogP applicability domain...")
        try:
            df = pd.read_csv(self.data_path)
            train_smiles = df['smiles'].tolist()
            
            self.ad = ApplicabilityDomain()
            self.ad.fit(train_smiles)
        except Exception as e:
             logger.error(f"Failed to fit LogP applicability domain: {e}")
             raise

    def predict(self, smiles):
        """
        Predict LogP for a single SMILES.
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
                     return {'error': f"Missing descriptor {col}"}
                desc_values.append(desc[col])
            
            fp_values = []
            for col in self.fingerprint_cols:
                 if col not in desc:
                     return {'error': "Missing fingerprint data"}
                 fp_values.append(desc[col])
            
            # Scaling
            desc_array = np.array(desc_values).reshape(1, -1)
            desc_scaled = self.scaler.transform(desc_array)
            
            fp_array = np.array(fp_values).reshape(1, -1)
            X_final = np.hstack([desc_scaled, fp_array])
            
            # XGB Prediction
            xgb_pred = float(self.xgb_model.predict(X_final)[0])
            
            # RF Uncertainty
            rf_res = self.uncertainty_estimator.predict_with_uncertainty(X_final)[0]
            rf_unc = rf_res['uncertainty_std']
            
            # Domain
            ad_res = self.ad.predict([smiles])[0]
            
            return {
                "prediction": xgb_pred,
                "uncertainty": rf_unc,
                "domain_status": "In Domain" if ad_res['in_domain'] else "Out of Domain"
            }

        except Exception as e:
            logger.error(f"LogP inference failed for {smiles}: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    engine = LogPInference()
    print(engine.predict("CCO"))
