from src.synthesis.synthesis_engine import SynthesisEngine
import logging

logging.basicConfig(level=logging.INFO)

def test_depth_two():
    engine = SynthesisEngine()
    # Aspirin
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    
    print(f"Testing depth=1 for {smiles}")
    res1 = engine.analyze(smiles, depth=1)
    print(f"Depth 1 Precursors: {len(res1['retrosynthesis']['precursors'])}")
    for p in res1['retrosynthesis']['precursors']:
        print(f" - {p['name']}: {p['smiles']} (Conf: {p['confidence']})")
        
    # N-acetyl-alanine methyl ester (Amide + Ester)
    smiles_complex = "CC(=O)NC(C)C(=O)OC"
    
    print(f"\nTesting depth=1 for {smiles_complex}")
    res1 = engine.analyze(smiles_complex, depth=1)
    print(f"Depth 1 Precursors: {len(res1['retrosynthesis']['precursors'])}")
    for p in res1['retrosynthesis']['precursors']:
        print(f" - {p['name']}: {p['smiles']} (Conf: {p['confidence']})")
        
    print(f"\nTesting depth=2 for {smiles_complex}")
    res2 = engine.analyze(smiles_complex, depth=2)
    print(f"Depth 2 Precursors: {len(res2['retrosynthesis']['precursors'])}")
    for p in res2['retrosynthesis']['precursors']:
        print(f" - {p['name']}: {p['smiles']} (Conf: {p['confidence']})")

if __name__ == "__main__":
    test_depth_two()
