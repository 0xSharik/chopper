
import os
import sys
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import rdMolDescriptors

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class DrugLikenessEngine:
    """Calculates drug-likeness scores based on molecular properties."""
    
    def __init__(self):
        pass
    
    def calculate(self, smiles):
        """
        Calculate drug-likeness score.
        
        Args:
            smiles: SMILES string
            
        Returns:
            dict: Drug-likeness metrics and overall score
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._error_result()
            
            # Calculate properties
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            rotatable_bonds = Lipinski.NumRotatableBonds(mol)
            
            # Lipinski Rule of 5
            lipinski_violations = 0
            if mw > 500:
                lipinski_violations += 1
            if logp > 5:
                lipinski_violations += 1
            if hbd > 5:
                lipinski_violations += 1
            if hba > 10:
                lipinski_violations += 1
            
            lipinski_pass = lipinski_violations <= 1
            
            # Veber's Rule
            veber_pass = (tpsa <= 140) and (rotatable_bonds <= 10)
            
            # Calculate overall score (0-100)
            score = 100.0
            
            # Lipinski penalties
            score -= lipinski_violations * 15
            
            # Veber penalties
            if not veber_pass:
                score -= 20
            
            # Additional penalties for extreme values
            if mw > 600:
                score -= 10
            if logp > 6 or logp < -2:
                score -= 10
            if tpsa > 160:
                score -= 10
            
            score = max(0, min(100, score))
            
            # Determine drug-likeness class
            if score >= 70:
                dl_class = "Good"
            elif score >= 50:
                dl_class = "Moderate"
            else:
                dl_class = "Poor"
            
            return {
                'score': float(score),
                'class': dl_class,
                'lipinski': {
                    'violations': lipinski_violations,
                    'pass': lipinski_pass,
                    'mw': float(mw),
                    'logp': float(logp),
                    'hbd': int(hbd),
                    'hba': int(hba)
                },
                'veber': {
                    'pass': veber_pass,
                    'tpsa': float(tpsa),
                    'rotatable_bonds': int(rotatable_bonds)
                }
            }
            
        except Exception as e:
            logger.error(f"Drug-likeness calculation failed: {e}")
            return self._error_result()
    
    def _error_result(self):
        """Return error result."""
        return {
            'score': None,
            'class': 'Unknown',
            'lipinski': {'violations': None, 'pass': False},
            'veber': {'pass': False}
        }

if __name__ == "__main__":
    # Test
    engine = DrugLikenessEngine()
    
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
    ]
    
    for smi in test_smiles:
        result = engine.calculate(smi)
        print(f"\nSMILES: {smi}")
        print(f"Score: {result['score']}")
        print(f"Class: {result['class']}")
        print(f"Lipinski: {result['lipinski']}")
        print(f"Veber: {result['veber']}")
