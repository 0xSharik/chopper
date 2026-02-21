import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.synthesis.synthesis_engine import SynthesisEngine

def test_forward_synthesis():
    engine = SynthesisEngine()
    
    # 1. Ester Test (Acid + Alcohol)
    print("\nRunning Esterification Test...")
    res = engine.virtual_synthesis(["CC(=O)O", "CCO"])
    products = res["product_details"]
    
    assert len(products) == 1, f"Expected 1 product for esterification, got {len(products)}"
    assert products[0]["reaction_name"] == "Ester formation"
    assert "admet_preview" in products[0]
    assert "classification" in products[0]
    print("  SUCCESS: Ester test passed.")

    # 2. Amide Test (Acid + Amine)
    print("\nRunning Amidation Test...")
    res = engine.virtual_synthesis(["CC(=O)O", "CCN"])
    products = res["product_details"]
    assert len(products) == 1
    assert products[0]["reaction_name"] == "Amide formation"
    print("  SUCCESS: Amide test passed.")

    # 3. Retrosynthesis / Analysis Test
    print("\nRunning Synthesis Analysis Test (Depth 1)...")
    analysis = engine.analyze("CC(=O)Oc1ccccc1C(=O)O", depth=1)
    assert analysis["retrosynthesis"]["confidence"] > 0
    print("  SUCCESS: Depth 1 passed.")

    print("\nRunning Synthesis Analysis Test (Depth 2)...")
    analysis2 = engine.analyze("CC(=O)Oc1ccccc1C(=O)O", depth=2)
    assert len(analysis2["retrosynthesis"]["precursors"]) >= len(analysis["retrosynthesis"]["precursors"])
    # Check if a precursor in depth 2 has a compound name (suggesting depth 2 worked)
    has_step2 = any("->" in p.get("name", "") for p in analysis2["retrosynthesis"]["precursors"])
    print(f"  SUCCESS: Depth 2 passed (Found step-2 disconnection: {has_step2}).")

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)

if __name__ == "__main__":
    try:
        test_forward_synthesis()
    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
