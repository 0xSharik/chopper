"""
SMILES editing and comparison utilities.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging

logger = logging.getLogger(__name__)

def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES string using RDKit.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        bool: True if valid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def canonicalize_smiles(smiles: str) -> str:
    """
    Return canonical SMILES representation.
    
    Args:
        smiles: Input SMILES
        
    Returns:
        str: Canonical SMILES or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except:
        return None

def compare_smiles(old_smiles: str, new_smiles: str):
    """
    Compare two SMILES strings and return structural differences.
    
    Args:
        old_smiles: Original SMILES
        new_smiles: Modified SMILES
        
    Returns:
        dict: Comparison metrics
    """
    try:
        old_mol = Chem.MolFromSmiles(old_smiles)
        new_mol = Chem.MolFromSmiles(new_smiles)
        
        if old_mol is None or new_mol is None:
            return None
        
        old_atoms = old_mol.GetNumAtoms()
        new_atoms = new_mol.GetNumAtoms()
        
        old_mw = Descriptors.MolWt(old_mol)
        new_mw = Descriptors.MolWt(new_mol)
        
        return {
            "atom_count_change": new_atoms - old_atoms,
            "mw_change": new_mw - old_mw,
            "old_atoms": old_atoms,
            "new_atoms": new_atoms,
            "old_mw": old_mw,
            "new_mw": new_mw
        }
    
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return None

def generate_edit_report(old_smiles: str, new_smiles: str):
    """
    Generate human-readable report of structural changes.
    
    Args:
        old_smiles: Original SMILES
        new_smiles: Modified SMILES
        
    Returns:
        str: Summary of changes
    """
    comparison = compare_smiles(old_smiles, new_smiles)
    
    if comparison is None:
        return "Unable to compare molecules"
    
    atom_change = comparison['atom_count_change']
    mw_change = comparison['mw_change']
    
    parts = []
    
    if atom_change > 0:
        parts.append(f"Added {atom_change} atom{'s' if atom_change > 1 else ''}")
    elif atom_change < 0:
        parts.append(f"Removed {abs(atom_change)} atom{'s' if abs(atom_change) > 1 else ''}")
    else:
        parts.append("Same atom count")
    
    if abs(mw_change) > 0.1:
        parts.append(f"MW changed by {mw_change:+.2f} Da")
    
    return ". ".join(parts) + "."

if __name__ == "__main__":
    # Test
    old = "c1ccccc1"  # Benzene
    new = "c1ccccc1O"  # Phenol
    
    print(f"Valid old: {validate_smiles(old)}")
    print(f"Valid new: {validate_smiles(new)}")
    
    print(f"\nCanonical old: {canonicalize_smiles(old)}")
    print(f"Canonical new: {canonicalize_smiles(new)}")
    
    comp = compare_smiles(old, new)
    print(f"\nComparison: {comp}")
    
    report = generate_edit_report(old, new)
    print(f"Report: {report}")
