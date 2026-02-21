import sys
import os
import json

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.formulation.mixture_parser import MixtureParser
from src.formulation.composition_engine import CompositionEngine
from src.formulation.multi_system_builder import MultiMoleculeSystemBuilder

def test_formulation_logic():
    print("="*60)
    print("Multi-Molecule Formulation Module Test")
    print("="*60)
    
    # 1. Test Parsing
    print("\n1. Testing Parser...")
    parser = MixtureParser()
    
    p1 = parser.parse("CCO, CC(=O)O")
    print(f"  String parse (comma): {p1}")
    
    p2 = parser.parse("CCO.CCO.CC(=O)O")
    print(f"  String parse (dot):   {p2}")
    
    p3 = parser.parse('[{"smiles": "CCO", "count": 10}, {"smiles": "CC(=O)O", "count": 5}]')
    print(f"  JSON parse:           {p3}")
    
    # 2. Test Composition Engine
    print("\n2. Testing Composition Engine...")
    comp_engine = CompositionEngine()
    
    raw = [{"smiles": "CCO", "count": 40}, {"smiles": "CC(=O)O", "count": 20}]
    normalized = comp_engine.normalize(raw, max_total=50)
    print(f"  Normalized (max 50): {normalized}")
    print(f"  Total count:         {sum(item['count'] for item in normalized)}")
    
    summary = comp_engine.get_summary(normalized)
    print(f"  Summary:             {summary['composition_pct']}")
    
    # 3. Test System Builder (Partial - check if it crashes during build setup)
    print("\n3. Testing System Builder (Logic Check)...")
    config = {"mode": "demo", "padding_nm": 1.2, "output_base_dir": "sim_output"}
    builder = MultiMoleculeSystemBuilder(config)
    
    mixture = [{"smiles": "CCO", "count": 2}, {"smiles": "CC(=O)O", "count": 1}]
    
    try:
        # We won't run the full build if it requires OpenMM installation/GPU access in this env,
        # but we can check if the conformer gen and XML gen steps work.
        print("  Generating unique XMLs...")
        for idx, item in enumerate(mixture):
             mol = builder.base_builder._smiles_to_3d(item['smiles'])
             props = builder.base_builder._prepare_molecule_for_mmff94(mol)
             xml = builder._generate_unique_solute_xml(mol, props, mol_idx=idx)
             print(f"    - XML generated for {item['smiles']} (Residue M{idx:02d})")
             
        print("\n  ✅ Core Logic Verification Passed")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    test_formulation_logic()
