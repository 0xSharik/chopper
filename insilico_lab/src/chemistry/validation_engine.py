"""
Molecule validation engine for safety checks.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging

logger = logging.getLogger(__name__)

def validate_modified_molecule(smiles: str):
    """
    Comprehensive validation of modified molecule.
    
    Checks:
    - RDKit sanitization
    - No radicals
    - No valency violations
    - MW < 2000 (relaxed for drug discovery)
    - Atom count < 300
    
    Args:
        smiles: SMILES to validate
        
    Returns:
        dict: {"valid": bool, "errors": list[str]}
    """
    errors = []
    
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            errors.append("Invalid SMILES - cannot parse")
            return {"valid": False, "errors": errors}
        
        # Sanitization check
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            errors.append(f"Sanitization failed: {str(e)}")
            return {"valid": False, "errors": errors}
        
        # Check for radicals
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                errors.append(f"Radical detected on atom {atom.GetIdx()}")
        
        # Check valency
        for atom in mol.GetAtoms():
            try:
                atom.GetTotalValence()
            except:
                errors.append(f"Valency violation on atom {atom.GetIdx()}")
        
        # Molecular weight check (relaxed for drug discovery)
        mw = Descriptors.MolWt(mol)
        if mw > 2000:
            errors.append(f"MW too high: {mw:.1f} > 2000")
        
        # Atom count check (relaxed)
        atom_count = mol.GetNumAtoms()
        if atom_count > 300:
            errors.append(f"Too many atoms: {atom_count} > 300")
        
        # Heavy atom check (relaxed)
        heavy_atoms = mol.GetNumHeavyAtoms()
        if heavy_atoms > 200:
            errors.append(f"Too many heavy atoms: {heavy_atoms} > 200")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "valid": False,
            "errors": [f"Validation exception: {str(e)}"]
        }

def quick_validate(smiles: str) -> bool:
    """
    Quick validation check.
    
    Args:
        smiles: SMILES string
        
    Returns:
        bool: True if valid
    """
    result = validate_modified_molecule(smiles)
    return result["valid"]

if __name__ == "__main__":
    # Test valid molecule
    valid = "c1ccccc1O"  # Phenol
    result = validate_modified_molecule(valid)
    print(f"Phenol validation: {result}")
    
    # Test invalid (too large)
    large = "C" * 200
    result = validate_modified_molecule(large)
    print(f"\nLarge molecule validation: {result}")
    
    # Test invalid SMILES
    invalid = "XYZ123"
    result = validate_modified_molecule(invalid)
    print(f"\nInvalid SMILES validation: {result}")
