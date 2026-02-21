import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import logging

logger = logging.getLogger(__name__)

try:
    # Try to import from traditional contribution if available
    from rdkit.Chem import RDConfig
    import os
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    _HAS_SASCORER = True
except ImportError:
    _HAS_SASCORER = False
    logger.info("RDKit sascorer not found. Using deterministic fallback.")

class SyntheticAccessibility:
    """Calculates Synthetic Accessibility (SA) Score."""
    
    def calculate(self, mol) -> float:
        """
        Calculate SA Score (1-10).
        1 = Easy to synthesize, 10 = Very difficult.
        """
        if _HAS_SASCORER:
            try:
                return round(sascorer.calculateScore(mol), 2)
            except Exception as e:
                logger.warning(f"Error in sascorer: {e}. Falling back.")
        
        return self._fallback_calculate(mol)

    def _fallback_calculate(self, mol) -> float:
        """Deterministic fallback for SA score."""
        # 1. Heavy atom penalty
        heavy_atoms = mol.GetNumHeavyAtoms()
        penalty_size = min(heavy_atoms / 10, 3)
        
        # 2. Ring count penalty
        ring_penalty = mol.GetRingInfo().NumRings() * 0.5
        
        # 3. Stereo center penalty
        stereo_penalty = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) * 0.5
        
        # 4. Fragment complexity (Morgan FP Density)
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        generator = GetMorganGenerator(radius=2, fpSize=1024)
        fp = generator.GetFingerprint(mol)
        on_bits = fp.GetNumOnBits()
        complexity_penalty = (on_bits / 1024) * 10 # Scale to 10
        
        # Combine components
        # Base score starts at 1.0 (easy)
        raw_score = 1.0 + penalty_size + ring_penalty + stereo_penalty + (complexity_penalty * 0.5)
        
        # Clamp between 1.0 and 10.0
        final_score = max(1.0, min(10.0, raw_score))
        
        return round(float(final_score), 2)
