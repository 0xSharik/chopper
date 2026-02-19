
import os
import sys
import logging
import numpy as np

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Correct relative imports if this file is run directly vs as module
# We should use absolute imports from src if sys.path is set correctly
from src.inference.esol_inference_engine import ESOLInference
from src.inference.logp_inference_engine import LogPInference
from src.inference.pka_inference_engine import pKaInference
from src.inference.permeability_engine import PermeabilityEngine
from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)

class PropertyEngine:
    def __init__(self, models_dir=None):
        """
        Initialize the Unified Property Engine.
        
        Args:
            models_dir (str, optional): Path to the directory containing all models.
                                        Defaults to 'models/' in project root.
        """
        if models_dir is None:
            models_dir = os.path.join(project_root, 'models')
            
        self.models_dir = models_dir
        self.esol_engine = ESOLInference(models_dir)
        self.logp_engine = LogPInference(models_dir)
        self.pka_engine = pKaInference(models_dir)
        self.perm_engine = PermeabilityEngine(models_dir)

    def _calculate_logd(self, logp, pka, ph=7.4):
        """
        Calculate logD at a specific pH using Henderson-Hasselbalch equation.
        Assuming monoprotic approximation based on pKa value.
        """
        try:
            pka = float(pka)
            logp = float(logp)
            
            if pka < 7.0: # Treat as acid
                term = 1 + 10**(ph - pka)
                logd = logp - np.log10(term)
            else: # Treat as base
                term = 1 + 10**(pka - ph)
                logd = logp - np.log10(term)
            
            return logd
        except:
            return None

    def _calculate_basic_descriptors(self, smiles):
        """Calculate basic RDKit descriptors."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            return {
                'MW': Descriptors.MolWt(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol)
            }
        except Exception as e:
            logger.error(f"Error calculating basic descriptors: {e}")
            return {}

    def predict_properties(self, smiles):
        """
        Predict all physicochemical properties for a molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            dict: All predicted properties
        """
        results = {}
        
        # Basic descriptors (always available from RDKit)
        basic = self._calculate_basic_descriptors(smiles)
        results.update(basic)
        
        # 1. Solubility (ESOL)
        try:
            esol_result = self.esol_engine.predict(smiles)
            if isinstance(esol_result, dict):
                # ESOL returns 'xgb_prediction' and 'rf_prediction', use XGB as primary
                results['logS'] = esol_result.get('xgb_prediction') or esol_result.get('rf_prediction')
            else:
                results['logS'] = esol_result
        except Exception as e:
            logger.warning(f"ESOL prediction failed: {e}")
            results['logS'] = None

        # 2. Lipophilicity (logP)
        try:
            logp_result = self.logp_engine.predict_logP(smiles)
            results['logP'] = logp_result.get('prediction') if isinstance(logp_result, dict) else logp_result
        except Exception as e:
            # Fallback: Use RDKit's Crippen logP
            try:
                from rdkit.Chem import Crippen
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    results['logP'] = Crippen.MolLogP(mol)
                else:
                    results['logP'] = None
            except:
                results['logP'] = None

        # 3. pKa - Currently disabled (model not trained with proper descriptors)
        # TODO: Retrain pKa model with descriptor features
        results['pKa'] = None

        # 4. logD (pH 7.4) Calculation
        try:
            logp_val = results.get('logP')
            pka_dict = results.get('pKa')
            
            if logp_val is not None and pka_dict and isinstance(pka_dict, dict):
                # Use acidic or basic pKa for calculation
                pka_val = pka_dict.get('acidic') or pka_dict.get('basic')
                if pka_val:
                    results['logD'] = self._calculate_logd(logp_val, pka_val, ph=7.4)
                else:
                    results['logD'] = None
            else:
                results['logD'] = None
        except:
            results['logD'] = None

        # 5. Permeability (Caco-2, BBB)
        try:
            perm_results = self.perm_engine.predict_permeability(smiles)
            results['permeability'] = perm_results
        except Exception as e:
            logger.warning(f"Permeability prediction failed: {e}")
            results['permeability'] = {}

        return results

if __name__ == "__main__":
    # Test
    engine = PropertyEngine()
    
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    print(f"Testing: {test_smiles}")
    
    results = engine.predict_properties(test_smiles)
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")
