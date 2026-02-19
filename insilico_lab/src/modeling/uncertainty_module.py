
import numpy as np
import logging
import joblib

# Setup logger
logger = logging.getLogger(__name__)

class UncertaintyEstimator:
    def __init__(self, model_path):
        """
        Initialize with a trained RandomForestRegressor.
        """
        self.model = self._load_model(model_path)
        
    def _load_model(self, path):
        try:
            model = joblib.load(path)
            if not hasattr(model, 'estimators_'):
                 raise ValueError("Model does not appear to be a RandomForest (no estimators_)")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise

    def predict_with_uncertainty(self, X):
        """
        Predict mean and standard deviation of tree predictions.
        
        Args:
            X (array-like): Input features.
            
        Returns:
            list of dicts: [{'prediction': mean, 'uncertainty_std': std}, ...]
        """
        try:
            # Collect predictions from all trees
            # shape: (n_estimators, n_samples)
            per_tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
            
            # Calculate stats
            means = np.mean(per_tree_preds, axis=0)
            stds = np.std(per_tree_preds, axis=0)
            
            results = []
            for m, s in zip(means, stds):
                results.append({
                    "prediction": float(m),
                    "uncertainty_std": float(s)
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error in uncertainty prediction: {e}")
            return []

if __name__ == "__main__":
    # Test script
    import os
    import pandas as pd
    import sys
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.append(project_root)
    
    from src.modeling.data_prep import load_and_prep_data
    
    logging.basicConfig(level=logging.INFO)
    
    model_path = os.path.join(project_root, 'models/esol_rf.joblib')
    data_path = os.path.join(project_root, 'data/processed/esol_features.csv')
    scaler_path = os.path.join(project_root, 'models/esol_scaler.joblib')
    
    if os.path.exists(model_path) and os.path.exists(data_path):
        ue = UncertaintyEstimator(model_path)
        
        # Load some data to test
        # Note: We need to scale descriptors properly!
        X, _, desc_cols, fp_cols = load_and_prep_data(data_path)
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        
        # Scale test sample
        sample_X = X.iloc[:5]
        sample_desc = sample_X[desc_cols]
        sample_desc_scaled = scaler.transform(sample_desc)
        sample_fp = sample_X[fp_cols].values
        
        sample_input = np.hstack([sample_desc_scaled, sample_fp])
        
        print("\n--- Uncertainty Test (First 5 samples) ---")
        preds = ue.predict_with_uncertainty(sample_input)
        for i, res in enumerate(preds):
            print(f"Sample {i}: Pred={res['prediction']:.4f}, Uncert={res['uncertainty_std']:.4f}")
