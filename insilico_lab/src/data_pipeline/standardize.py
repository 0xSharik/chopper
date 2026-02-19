import logging
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Configure logger
logger = logging.getLogger(__name__)

def smiles_to_mol(smiles):
    """
    Convert SMILES string to RDKit Mol object.
    
    Args:
        smiles (str): SMILES string.
        
    Returns:
        rdkit.Chem.rdchem.Mol: RDKit Mol object, or None if invalid.
    """
    try:
        if not isinstance(smiles, str):
            logger.warning(f"Input is not a string: {smiles}")
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return None
        return mol
    except Exception as e:
        logger.error(f"Error converting SMILES '{smiles}': {e}")
        return None

def remove_salts(mol):
    """
    Remove salts and small fragments from the molecule.
    Uses rdMolStandardize.FragmentParent logic.
    
    Args:
        mol (rdkit.Chem.rdchem.Mol): Input molecule.
        
    Returns:
        rdkit.Chem.rdchem.Mol: Desalted molecule.
        
    Raises:
        ValueError: If input mol is None.
    """
    if mol is None:
        raise ValueError("Input molecule is None")
        
    try:
        # FragmentParent returns the largest organic fragment
        clean_mol = rdMolStandardize.FragmentParent(mol)
        return clean_mol
    except Exception as e:
        logger.error(f"Error removing salts: {e}")
        # Return original mol if standardization fails? Or raise?
        # Usually better to return None or raise to indicate failure in pipeline
        # But per requirements, let's log and return None for safety in pipeline
        return None

def canonicalize_smiles(mol):
    """
    Generate canonical SMILES from a molecule.
    
    Args:
        mol (rdkit.Chem.rdchem.Mol): Input molecule.
        
    Returns:
        str: Canonical SMILES string.
    """
    if mol is None:
        return None
        
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) # Commonly non-isomeric for simple canonicalization, but default canonical=True handles it. 
        # Requirement says "Return canonical SMILES". Default is usually fine.
    except Exception as e:
        logger.error(f"Error canonicalizing molecule: {e}")
        return None

def standardize_smiles(smiles):
    """
    Full standardization pipeline:
    1. Convert to Mol
    2. Remove salts
    3. Canonicalize
    
    Args:
        smiles (str): Input SMILES string.
        
    Returns:
        str: Standardized SMILES string, or None if failure.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
        
    try:
        clean_mol = remove_salts(mol)
        if clean_mol is None:
            logger.warning(f"Salt removal failed for SMILES: {smiles}")
            return None
            
        std_smiles = canonicalize_smiles(clean_mol)
        if std_smiles is None:
            logger.warning(f"Canonicalization failed for SMILES: {smiles}")
            return None
            
        return std_smiles
    except Exception as e:
        logger.error(f"Standardization pipeline failed for '{smiles}': {e}")
        return None

if __name__ == "__main__":
    # Configure logging for standalone testing
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    
    test_smiles = "CN(C)C.Cl" # Salt example
    print(f"Original: {test_smiles}")
    std = standardize_smiles(test_smiles)
    print(f"Standardized: {std}")

