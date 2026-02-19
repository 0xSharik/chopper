"""
Reaction engine for attaching functional groups to molecules.
"""
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from src.chemistry.group_library import get_group_smiles
from src.chemistry.atom_tools import is_valid_attachment_point, get_attachable_atoms

logger = logging.getLogger(__name__)

def attach_group(smiles: str, group_name: str, atom_index: int = None):
    """
    Attach a functional group to a molecule.
    
    Args:
        smiles: Base molecule SMILES
        group_name: Name of functional group from library
        atom_index: Atom index for attachment (None = auto-select first)
        
    Returns:
        str: Modified SMILES or None if failed
    """
    try:
        # Load base molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error("Invalid base SMILES")
            return None
        
        # Get group SMILES
        group_smiles = get_group_smiles(group_name)
        if group_smiles is None:
            logger.error(f"Unknown group: {group_name}")
            return None
        
        # Auto-select atom if not specified
        if atom_index is None:
            attachable = get_attachable_atoms(smiles)
            if not attachable:
                logger.error("No attachable atoms found")
                return None
            atom_index = attachable[0]
        
        # Validate attachment point
        if not is_valid_attachment_point(smiles, atom_index):
            logger.error(f"Invalid attachment point: {atom_index}")
            return None
        
        # Use advanced attachment with EditableMol
        return attach_group_advanced(smiles, group_smiles, atom_index)
    
    except Exception as e:
        logger.error(f"Attachment failed: {e}")
        return None

def attach_group_advanced(smiles: str, group_smiles: str, atom_index: int):
    """
    Advanced attachment using RDKit reaction SMARTS.
    
    Args:
        smiles: Base molecule
        group_smiles: Group to attach
        atom_index: Attachment point
        
    Returns:
        str: Modified SMILES or None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        group = Chem.MolFromSmiles(group_smiles)
        
        if mol is None or group is None:
            return None
        
        # Use EditableMol for precise control
        em = Chem.EditableMol(mol)
        
        # Add atoms from group
        atom_map = {}
        for atom in group.GetAtoms():
            new_idx = em.AddAtom(atom)
            atom_map[atom.GetIdx()] = new_idx
        
        # Add bonds from group
        for bond in group.GetBonds():
            em.AddBond(
                atom_map[bond.GetBeginAtomIdx()],
                atom_map[bond.GetEndAtomIdx()],
                bond.GetBondType()
            )
        
        # Connect to base molecule
        first_group_atom = atom_map[0]
        em.AddBond(atom_index, first_group_atom, Chem.BondType.SINGLE)
        
        # Get modified molecule
        new_mol = em.GetMol()
        
        # Sanitize
        Chem.SanitizeMol(new_mol)
        
        return Chem.MolToSmiles(new_mol)
    
    except Exception as e:
        logger.error(f"Advanced attachment failed: {e}")
        return None

if __name__ == "__main__":
    # Test attachment
    base = "c1ccccc1"  # Benzene
    print(f"Base: {base}")
    
    result = attach_group(base, "methyl")
    print(f"After adding methyl: {result}")
    
    result = attach_group(base, "hydroxyl")
    print(f"After adding hydroxyl: {result}")
