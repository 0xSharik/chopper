
import logging
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, AllChem, Crippen

# Configure logger
logger = logging.getLogger(__name__)

def calculate_descriptors(smiles):
    """
    Calculate molecular descriptors for a given SMILES string.
    
    Descriptors:
    - Molecular Weight (MW)
    - Topological Polar Surface Area (TPSA)
    - Number of H-Bond Donors (HBD)
    - Number of H-Bond Acceptors (HBA)
    - Number of Rotatable Bonds (RotBonds)
    - Aromatic Ring Count (AroRings)
    - Crippen logP (LogP)
    - Morgan Fingerprint (radius=2, nBits=2048) -> returned as bit vector list/array
    
    Args:
        smiles (str): Canonical SMILES string.
        
    Returns:
        dict: Dictionary of descriptors, or None if invalid.
              Fingerprint is returned as a list of integers (0/1).
    """
    try:
        if not isinstance(smiles, str):
            logger.warning(f"Invalid input: {smiles}")
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES for descriptors: {smiles}")
            return None

        # Calculate physico-chemical descriptors
        descriptors = {
            'MW': Descriptors.MolWt(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'RotBonds': Lipinski.NumRotatableBonds(mol),
            'AroRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'LogP': Crippen.MolLogP(mol)
        }
        
        # Calculate Morgan Fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        # Convert to numpy array or list for easier handling downstream (e.g. sklearn)
        # Keeping as list for JSON serialization if needed, or returning as object
        # The prompt asks for "dictionary or feature vector". 
        # Making it flat is often easier for ML. 
        # But let's keep structured for now: descriptors + fp
        descriptors['MorganFP'] = list(fp)
        
        return descriptors

    except Exception as e:
        logger.error(f"Error calculating descriptors for '{smiles}': {e}")
        return None

def get_descriptor_names():
    """Returns list of descriptor names (excluding fingerprint breakdown)."""
    return ['MW', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'AroRings', 'LogP']

if __name__ == "__main__":
    # Test execution
    test_smiles = "CCO"
    print(f"Calculating descriptors for: {test_smiles}")
    desc = calculate_descriptors(test_smiles)
    if desc:
        print("Descriptors calculated successfully:")
        for k, v in desc.items():
            if k == 'MorganFP':
                print(f"{k}: <vector of length {len(v)}>")
            else:
                print(f"{k}: {v}")
    else:
        print("Failed to calculate descriptors.")
