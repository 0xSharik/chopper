
import numpy as np
import logging
import joblib
import pandas as pd
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem

# Setup logger
logger = logging.getLogger(__name__)

class ApplicabilityDomain:
    def __init__(self, fingerprint_type='morgan', radius=2, nBits=2048):
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.nBits = nBits
        self.training_fps = []
        
    def fit(self, smiles_list):
        """
        Store fingerprints of training data.
        """
        self.training_fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.nBits)
                self.training_fps.append(fp)
        logger.info(f"Fitted applicability domain with {len(self.training_fps)} fingerprints.")

    def predict(self, smiles_list, threshold=0.3):
        """
        Calculate similarity to training set.
        
        Returns:
            list of dicts: {'max_similarity': float, 'in_domain': bool}
        """
        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                results.append({'max_similarity': 0.0, 'in_domain': False})
                continue
                
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.nBits)
            
            # BulkTanimotoSimilarity is faster
            sims = DataStructs.BulkTanimotoSimilarity(fp, self.training_fps)
            max_sim = max(sims) if sims else 0.0
            
            results.append({
                'max_similarity': max_sim,
                'in_domain': max_sim >= threshold
            })
            
        return results

if __name__ == "__main__":
    # Test
    import os
    import sys
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, 'data/processed/esol_features.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Split fit/test to demo
        train_smiles = df['smiles'].iloc[:500].tolist()
        test_smiles = df['smiles'].iloc[500:505].tolist()
        
        ad = ApplicabilityDomain()
        ad.fit(train_smiles)
        
        print("\n--- Applicability Domain Test ---")
        preds = ad.predict(test_smiles)
        for i, res in enumerate(preds):
            print(f"Sample {i}: Max Sim={res['max_similarity']:.4f}, In Domain={res['in_domain']}")
