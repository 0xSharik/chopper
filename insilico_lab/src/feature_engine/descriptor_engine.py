"""
Unified descriptor generation engine.
Generates molecular descriptors and fingerprints for all models.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Lipinski, Crippen, GraphDescriptors
import numpy as np
import logging

# Configure logger
logger = logging.getLogger(__name__)

def generate_descriptors(smiles):
    """
    Calculate molecular descriptors for a given SMILES string.
    
    Descriptors:
    - Molecular Weight (MW)
    - Topological Polar Surface Area (TPSA)
    - Number of H-Bond Donors (HBD)
    - Number of H-Bond Acceptors (HBA)
    - Number of Rotatable Bonds (RotBonds)
    - Aromatic Ring Count (AroRings)
    - Ring Count (RingCount)
    - Heavy Atom Count (HeavyAtomCount)
    - FractionCSP3 (FractionCSP3)
    - Crippen logP (LogP)
    - Ionization descriptors (FormalCharge, NumValenceElectrons)
    - Advanced ADME descriptors (MolVolume, ExactMW, Qed)
    - Meta-interactions (HBD/HBA, LogP*TPSA, etc)
    - Gasteiger Charges
    - Lipinski Violations
    - Morgan Fingerprint (radius=2, nBits=2048) -> bit vector components
    
    Args:
        smiles (str): Canonical SMILES string.
        
    Returns:
        dict: Dictionary of descriptors and fingerprint bits.
              Returns None if invalid.
    """
    try:
        if not isinstance(smiles, str):
            logger.warning(f"Invalid input: {smiles}")
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return None

        # Basic Descriptors
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'NumHDonors': Lipinski.NumHDonors(mol),
            'NumHAcceptors': Lipinski.NumHAcceptors(mol),
            'TPSA': rdMolDescriptors.CalcTPSA(mol),
            'NumAromaticRings': Lipinski.NumAromaticRings(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),  # Fixed: was 'RotatableBonds'
            'FractionCSP3': Lipinski.FractionCSP3(mol),
            'NumHeteroatoms': Lipinski.NumHeteroatoms(mol),
            'NumRings': Lipinski.RingCount(mol),
            'MolMR': Crippen.MolMR(mol),
            'BertzCT': GraphDescriptors.BertzCT(mol),
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
            'AtomCount': mol.GetNumAtoms(),
            
            # Ionization (Stage 4)
            'FormalCharge': Chem.GetFormalCharge(mol),
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
            
            # Stage 5: Advanced ADME Descriptors
            'MolVolume': Descriptors.ExactMolWt(mol), # Approx via ExactMW for now
            'ExactMW': Descriptors.ExactMolWt(mol),
            'Qed': Descriptors.qed(mol),
            
            # Meta-interactions (Cross-property)
            'HBD_HBA_Ratio': Descriptors.NumHDonors(mol) / (Descriptors.NumHAcceptors(mol) + 1e-6),
            'LogP_TPSA': Descriptors.MolLogP(mol) * Descriptors.TPSA(mol),
            'MW_LogP': Descriptors.MolWt(mol) * Descriptors.MolLogP(mol),
            'HBD_TPSA': Descriptors.NumHDonors(mol) * Descriptors.TPSA(mol),
        }
        
        # Add aliases for backward compatibility with trained models
        descriptors['MW'] = descriptors['MolWt']
        descriptors['HBD'] = descriptors['NumHDonors']
        descriptors['HBA'] = descriptors['NumHAcceptors']
        descriptors['RotatableBonds'] = descriptors['RotBonds']
        descriptors['AromaticRings'] = descriptors['NumAromaticRings']
        descriptors['AroRings'] = descriptors['NumAromaticRings']  # ESOL uses this name
        descriptors['RingCount'] = descriptors['NumRings']


        # Gasteiger Charges (requires computation)
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms() if atom.HasProp('_GasteigerCharge')]
            if charges:
                descriptors['MaxAbsPartialCharge'] = max(abs(c) for c in charges)
                descriptors['MaxPartialCharge'] = max(charges)
                descriptors['MinPartialCharge'] = min(charges)
            else:
                descriptors['MaxAbsPartialCharge'] = 0.0
                descriptors['MaxPartialCharge'] = 0.0
                descriptors['MinPartialCharge'] = 0.0
        except:
             descriptors['MaxAbsPartialCharge'] = 0.0
             descriptors['MaxPartialCharge'] = 0.0
             descriptors['MinPartialCharge'] = 0.0

        # Lipinski Violations
        violations = 0
        if descriptors['MW'] > 500: violations += 1
        if descriptors['LogP'] > 5: violations += 1
        if descriptors['HBD'] > 5: violations += 1
        if descriptors['HBA'] > 10: violations += 1
        descriptors['LipinskiViolations'] = violations
        
        # Calculate Morgan Fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        # Convert to dictionary with prefix 'fp_'
        fp_dict = {f'fp_{i}': int(fp.GetBit(i)) for i in range(2048)}
        
        # Merge descriptors and fingerprint
        features = {**descriptors, **fp_dict}
        
        return features

    except Exception as e:
        logger.error(f"Error calculating descriptors for '{smiles}': {e}")
        return None

if __name__ == "__main__":
    # Internal test
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    desc = generate_descriptors("CCO")
    if desc:
        print(f"Descriptors generated: {len(desc)} keys")
        print(f"MW: {desc['MW']}")
        print(f"TPSA: {desc['TPSA']}")
