
"""
reproduce_sasa_issue.py — Verify SASA fix for 1,1,2,2-tetrachloroethane.
"""
import os
import shutil
from src.md.md_engine import MolecularDynamicsEngine

def test_sasa():
    print("🧪 Verifying SASA Fix for Tetrachloroethane...")
    
    # SMILES for 1,1,2,2-tetrachloroethane
    smiles = "ClC(Cl)C(Cl)Cl"
    
    config = {
        "mode": "demo",
        "output_base_dir": "data/test_sasa",
        "production_ns": 0.002, # very short
        "equilibration_nvt_ps": 1.0,
        "equilibration_npt_ps": 1.0,
        "energy_minimization_steps": 100
    }
    
    if os.path.exists(config["output_base_dir"]):
        shutil.rmtree(config["output_base_dir"])
        
    engine = MolecularDynamicsEngine(config=config)
    result = engine.run(smiles, molecule_id="tetrachloroethane_test")
    
    if result["status"] == "success":
        sasa = result["metrics"]["sasa_mean"]
        print(f"✅ Simulation Success. SASA: {sasa:.2f} nm²")
        
        if sasa > 100.0:
            print("❌ FAILURE: SASA is still unreasonably high (likely including solvent).")
            exit(1)
        elif sasa < 1.0:
             print("⚠️ WARNING: SASA seems too low?")
        else:
            print("✅ PASS: SASA is in reasonable range for a small molecule.")
    else:
        print(f"❌ Simulation Failed: {result.get('message')}")
        exit(1)

if __name__ == "__main__":
    test_sasa()
