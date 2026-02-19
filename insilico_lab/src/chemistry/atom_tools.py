"""
Atom-level utilities for molecule editing.
"""
from rdkit import Chem
import logging

logger = logging.getLogger(__name__)

def get_attachable_atoms(smiles: str):
    """
    Get list of atom indices that can accept new attachments.
    
    Criteria:
    - Has at least one implicit hydrogen (can be replaced)
    - Not aromatic nitrogen (too unstable)
    - For attachment, we replace an H with the new group
    
    Args:
        smiles: SMILES string
        
    Returns:
        list[int]: Atom indices suitable for attachment
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # Sanitize to ensure proper valence calculation
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        
        attachable = []
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            
            # Get total number of hydrogens (explicit + implicit)
            num_h = atom.GetTotalNumHs()
            
            # If no hydrogens, cannot attach (nothing to replace)
            if num_h == 0:
                continue
            
            # Skip aromatic nitrogens (unstable when substituted)
            if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                continue
            
            # Any atom with at least one H can have that H replaced
            # This works for both saturated (replace H) and unsaturated (add bond)
            attachable.append(idx)
        
        return attachable
    
    except Exception as e:
        logger.error(f"Error getting attachable atoms: {e}")
        return []

def get_atom_summary(smiles: str):
    """
    Get detailed summary of all atoms in molecule.
    
    Args:
        smiles: SMILES string
        
    Returns:
        list[dict]: Atom information
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        summary = []
        
        for atom in mol.GetAtoms():
            info = {
                "index": atom.GetIdx(),
                "symbol": atom.GetSymbol(),
                "degree": atom.GetDegree(),
                "implicit_h": atom.GetTotalNumHs(),
                "aromatic": atom.GetIsAromatic(),
                "formal_charge": atom.GetFormalCharge()
            }
            summary.append(info)
        
        return summary
    
    except Exception as e:
        logger.error(f"Error getting atom summary: {e}")
        return []

def is_valid_attachment_point(smiles: str, atom_index: int):
    """
    Check if specific atom can accept attachment.
    
    Args:
        smiles: SMILES string
        atom_index: Atom index to check
        
    Returns:
        bool: True if valid attachment point
    """
    attachable = get_attachable_atoms(smiles)
    return atom_index in attachable

if __name__ == "__main__":
    # Test with various molecules
    test_cases = [
        "c1ccccc1",  # Benzene
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "CCO",  # Ethanol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    ]
    
    for test_smiles in test_cases:
        print(f"\nTesting: {test_smiles}")
        
        attachable = get_attachable_atoms(test_smiles)
        print(f"Attachable atoms: {attachable}")
        
        summary = get_atom_summary(test_smiles)
        print(f"Total atoms: {len(summary)}, Attachable: {len(attachable)}")
