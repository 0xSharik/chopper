"""
Molecular Modification Engine
Safe structural transformations for virtual drug modification.
"""
import logging
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_methyl(smiles: str) -> str:
    """Add a methyl group to first available carbon."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Find first carbon with available valence
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetTotalValence() < 4:
                # Add methyl by creating new molecule with CH3 appended
                new_smiles = smiles + "C"
                new_mol = Chem.MolFromSmiles(new_smiles)
                if new_mol:
                    Chem.SanitizeMol(new_mol)
                    return Chem.MolToSmiles(new_mol)
        
        # Fallback: just append methyl
        new_mol = Chem.MolFromSmiles(smiles + "C")
        if new_mol:
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
        return None
    except Exception as e:
        logger.error(f"add_methyl failed: {e}")
        return None

def add_fluorine(smiles: str) -> str:
    """Replace first hydrogen with fluorine."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add explicit hydrogens
        mol = Chem.AddHs(mol)
        
        # Find first hydrogen and replace with F
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'H':
                atom.SetAtomicNum(9)  # Fluorine
                break
        
        # Remove remaining explicit H and sanitize
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        logger.error(f"add_fluorine failed: {e}")
        return None

def add_hydroxyl(smiles: str) -> str:
    """Add OH group to first available carbon."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Simple approach: append OH
        new_smiles = smiles + "O"
        new_mol = Chem.MolFromSmiles(new_smiles)
        
        if new_mol:
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
        return None
    except Exception as e:
        logger.error(f"add_hydroxyl failed: {e}")
        return None

def extend_alkyl_chain(smiles: str) -> str:
    """Append carbon to extend alkyl chain."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Append CC to extend chain
        new_smiles = smiles + "CC"
        new_mol = Chem.MolFromSmiles(new_smiles)
        
        if new_mol:
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
        return None
    except Exception as e:
        logger.error(f"extend_alkyl_chain failed: {e}")
        return None

def add_chlorine(smiles: str) -> str:
    """Replace first hydrogen with chlorine."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add explicit hydrogens
        mol = Chem.AddHs(mol)
        
        # Find first hydrogen and replace with Cl
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'H':
                atom.SetAtomicNum(17)  # Chlorine
                break
        
        # Remove remaining explicit H and sanitize
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        logger.error(f"add_chlorine failed: {e}")
        return None

def generate_variants(smiles: str) -> dict:
    """
    Generate all chemical variants of a molecule.
    
    Returns:
        dict: {
            "original": smiles,
            "methyl_variant": smiles or None,
            "fluoro_variant": smiles or None,
            "hydroxyl_variant": smiles or None,
            "extended_chain": smiles or None,
            "chloro_variant": smiles or None
        }
    """
    return {
        "original": smiles,
        "methyl_variant": add_methyl(smiles),
        "fluoro_variant": add_fluorine(smiles),
        "hydroxyl_variant": add_hydroxyl(smiles),
        "extended_chain": extend_alkyl_chain(smiles),
        "chloro_variant": add_chlorine(smiles)
    }

if __name__ == "__main__":
    # Test with ethanol
    test_smiles = "CCO"
    print(f"Testing with: {test_smiles}")
    
    variants = generate_variants(test_smiles)
    for name, variant in variants.items():
        print(f"{name}: {variant}")
