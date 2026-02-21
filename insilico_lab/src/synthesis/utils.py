from rdkit import Chem

def validate_smiles(smiles: str):
    """Validate and sanitize SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    Chem.SanitizeMol(mol)
    return mol

def canonicalize(smiles: str):
    """Return canonical SMILES."""
    mol = validate_smiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True)

def remove_duplicates(smiles_list):
    """Remove duplicate SMILES from a list."""
    return list(set(smiles_list))
