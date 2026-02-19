
"""
validation.py — Structure validation for the Design & Simulate workflow.
"""
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger("design_workflow")

def validate_smiles(smiles: str) -> bool:
    """
    Validate a SMILES string for simulation readiness.
    
    Checks:
    1. RDKit sanitization
    2. No valence errors
    3. No disconnected fragments
    4. Minimum heavy atoms >= 5
    5. Net charge magnitude check
    
    Returns:
        True if valid, raises ValueError otherwise.
    """
    if not smiles:
        raise ValueError("SMILES string is empty.")
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
        
    # 1. Sanitization (caught by MolFromSmiles usually, but explicit check good)
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise ValueError(f"Sanitization failed: {e}")
        
    # 3. Disconnected fragments
    frags = Chem.GetMolFrags(mol)
    if len(frags) > 1:
        raise ValueError(f"Molecule has {len(frags)} disconnected fragments. Simulation requires a single connected component.")
        
    # 4. Minimum heavy atoms
    num_heavy = mol.GetNumHeavyAtoms()
    if num_heavy < 5:
        raise ValueError(f"Molecule too small ({num_heavy} heavy atoms). Minimum 5 required for meaningful MD.")
        
    # 5. Charge check
    charge = Chem.GetFormalCharge(mol)
    if abs(charge) > 2:
        logger.warning(f"High net charge ({charge}). Simulation determines solvated behavior, but high charge may require careful parameterized ions.")
        
    return True

def check_modification_reasonable(parent_smiles: str, variant_smiles: str) -> bool:
    """
    Check if the modification is chemically reasonable relative to the parent.
    
    Checks:
    1. Atom count difference < 10
    2. Ring count preservation (warning)
    
    Returns:
        True if reasonable, warning logged if suspicious.
    """
    p_mol = Chem.MolFromSmiles(parent_smiles)
    v_mol = Chem.MolFromSmiles(variant_smiles)
    
    if not p_mol or not v_mol:
        return False
        
    # 1. Atom count diff
    p_atoms = p_mol.GetNumHeavyAtoms()
    v_atoms = v_mol.GetNumHeavyAtoms()
    
    diff = abs(v_atoms - p_atoms)
    if diff >= 10:
        logger.warning(f"Large structural change detected: {diff} heavy atoms difference.")
        
    # 2. Ring count
    p_rings = Descriptors.RingCount(p_mol)
    v_rings = Descriptors.RingCount(v_mol)
    
    if p_rings != v_rings:
        logger.warning(f"Ring count changed from {p_rings} to {v_rings}.")
        
    return True
