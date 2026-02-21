import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.synthesis.synthesis_engine import SynthesisEngine

def test_synthesis():
    engine = SynthesisEngine()
    
    test_molecules = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Nitroglycerin": "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]"
    }
    
    print("="*60)
    print("Synthesis Intelligence Module Test")
    print("="*60)
    
    for name, smiles in test_molecules.items():
        print(f"\nAnalyzing: {name}")
        print(f"SMILES:    {smiles}")
        
        try:
            result = engine.analyze(smiles)
            print(f"  SAS Score:          {result['sas_score']}")
            print(f"  Ring Strain:        {result['ring_strain_score']}")
            print(f"  Functional Penalty: {result['functional_penalty_score']}")
            print(f"  Feasibility Score:  {result['feasibility_score']}%")
            print(f"  Classification:     {result['classification']}")
            print(f"  Precursors found:   {len(result['precursors'])}")
            for i, p in enumerate(result['precursors']):
                print(f"    [{i+1}] {p}")
                
        except Exception as e:
            print(f"  FAILED: {e}")
            
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    test_synthesis()
