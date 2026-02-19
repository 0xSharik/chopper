
from .esol_inference_engine import ESOLInference

_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        _engine = ESOLInference()
    return _engine

def predict_logS(smiles):
    """
    Predict solubility (logS) for a given SMILES string.
    
    Args:
        smiles (str): SMILES string of the molecule.
        
    Returns:
        dict: {
            "smiles": str,
            "xgb_prediction": float, (Recommended)
            "rf_prediction": float,
            "rf_uncertainty": float,
            "max_similarity": float,
            "domain_status": str ("In Domain" / "Out of Domain")
        }
    """
    engine = _get_engine()
    return engine.predict(smiles)
