"""
Functional Group Library
Curated attachable functional groups for medicinal chemistry.
"""
from rdkit import Chem

FUNCTIONAL_GROUPS = {
    "methyl": "[CH3]",
    "ethyl": "CC",
    "propyl": "CCC",
    "fluoro": "F",
    "chloro": "Cl",
    "bromo": "Br",
    "iodo": "I",
    "hydroxyl": "O",
    "amino": "N",
    "carboxyl": "C(=O)O",
    "nitro": "[N+](=O)[O-]",
    "trifluoromethyl": "C(F)(F)F",
    "methoxy": "OC",
    "ethoxy": "OCC",
    "amide": "C(=O)N",
    "cyano": "C#N",
    "sulfonyl": "S(=O)(=O)",
    "phenyl": "c1ccccc1",
    "acetyl": "C(=O)C",
    "isopropyl": "C(C)C",
    "tert-butyl": "C(C)(C)C"
}

def get_group(name: str):
    """
    Get RDKit molecule object for a functional group.
    
    Args:
        name: Name of functional group
        
    Returns:
        RDKit Mol object or None if invalid
    """
    smiles = FUNCTIONAL_GROUPS.get(name.lower())
    if smiles is None:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None

def list_groups():
    """
    Get list of available functional group names.
    
    Returns:
        list[str]: Sorted list of group names
    """
    return sorted(FUNCTIONAL_GROUPS.keys())

def get_group_smiles(name: str):
    """
    Get SMILES string for a functional group.
    
    Args:
        name: Name of functional group
        
    Returns:
        str: SMILES or None if invalid
    """
    return FUNCTIONAL_GROUPS.get(name.lower())

if __name__ == "__main__":
    print("Available functional groups:")
    for group in list_groups():
        smiles = get_group_smiles(group)
        print(f"  {group}: {smiles}")
