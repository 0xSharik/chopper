
import pandas as pd
import numpy as np
import os
import sys
import logging
import joblib
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem

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

class ESOLInference:
    def __init__(self, model_dir=None, data_path=None):
        if model_dir is None:
            model_dir = os.path.join(project_root, 'models')
        if data_path is None:
            data_path = os.path.join(project_root, 'data/processed/esol_features.csv')
            
        self.xgb_path = os.path.join(model_dir, 'esol_xgb.joblib')
        self.rf_path = os.path.join(model_dir, 'esol_rf.joblib')
        self.scaler_path = os.path.join(model_dir, 'esol_scaler.joblib')
        self.data_path = data_path
        
        self._load_artifacts()
        self._fit_domain()

    def _load_artifacts(self):
        logger.info("Loading models and scaler...")
        try:
            self.xgb_model = joblib.load(self.xgb_path)
            self.scaler = joblib.load(self.scaler_path)
            
            # Uncertainty estimator handles RF loading
            self.uncertainty_estimator = UncertaintyEstimator(self.rf_path)
            
            # Get column names to ensure correct order
            # We strictly need to know which are descriptors vs fingerprints
            # We can re-derive this from data_prep or hardcode based on engine
            # Better to re-derive to stay synced with training
            _, _, self.descriptor_cols, self.fingerprint_cols = load_and_prep_data(self.data_path)
            
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise

    def _fit_domain(self):
        logger.info("Fitting applicability domain...")
        try:
            # Load training SMILES
            df = pd.read_csv(self.data_path)
            train_smiles = df['smiles'].tolist()
            
            self.ad = ApplicabilityDomain()
            self.ad.fit(train_smiles)
        except Exception as e:
             logger.error(f"Failed to fit applicability domain: {e}")
             raise

    def predict(self, smiles):
        """
        Run full inference pipeline for a single SMILES.
        """
        try:
            # 1. Validation
            if not smiles or not isinstance(smiles, str):
                return {'error': 'Invalid SMILES'}

            # 2. Features
            desc = generate_descriptors(smiles)
            if desc is None:
                return {'error': 'Descriptor generation failed'}
                
            # Dictionary to DataFrame for scaling
            # Must ensure column order matches training!
            
            # Extract descriptors
            desc_values = []
            for col in self.descriptor_cols:
                if col not in desc:
                     logger.error(f"Missing descriptor: {col}")
                     return {'error': f"Missing descriptor {col}"}
                desc_values.append(desc[col])
            
            # Extract fingerprints (fp_0 ... fp_2047)
            fp_values = []
            # Check length implicitly by iterating cols
            for col in self.fingerprint_cols:
                 if col not in desc:
                     logger.error(f"Missing fingerprint bit: {col}")
                     return {'error': "Missing fingerprint data"}
                 fp_values.append(desc[col])

            if len(fp_values) != 2048:
                 return {'error': f"Fingerprint length mismatch: {len(fp_values)}"}
            
            # 3. Scaling
            # Reshape for scaler (1 sample)
            desc_array = np.array(desc_values).reshape(1, -1)
            desc_scaled = self.scaler.transform(desc_array)
            
            # Combine
            fp_array = np.array(fp_values).reshape(1, -1)
            X_final = np.hstack([desc_scaled, fp_array])
            
            # 4. Prediction
            
            # XGBoost
            xgb_pred = float(self.xgb_model.predict(X_final)[0])
            
            # RF + Uncertainty
            rf_res = self.uncertainty_estimator.predict_with_uncertainty(X_final)[0]
            rf_pred = rf_res['prediction']
            rf_unc = rf_res['uncertainty_std']
            
            # 5. Applicability Domain
            ad_res = self.ad.predict([smiles])[0]
            
            return {
                "smiles": smiles,
                "xgb_prediction": xgb_pred,
                "rf_prediction": rf_pred,
                "rf_uncertainty": rf_unc,
                "max_similarity": ad_res['max_similarity'],
                "domain_status": "In Domain" if ad_res['in_domain'] else "Out of Domain"
            }

        except Exception as e:
            logger.error(f"Inference failed for {smiles}: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Quick test
    engine = ESOLInference()
    res = engine.predict("CCO")
    print(res)
