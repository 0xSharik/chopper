"""
verify_research_md.py — Verification script for the Research-Grade MD Engine.

Runs a short "Demo" mode simulation on Aspirin to verify:
1. System Building (MMFF94 + TIP3P)
2. Equilibration steps
3. Production run
4. Analysis & Validation (Score generation)
5. Output file generation (JSON report, DCD, PDB)
"""

import os
import sys
import shutil

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.md.md_engine import MolecularDynamicsEngine

def verify():
    print("🔬 Verifying Research-Grade MD Engine...")
    
    # Configure for ultra-fast demo verification
    config = {
        "mode": "demo",
        "output_base_dir": "data/md_runs_verify",
        "production_ns": 0.05, # Short but reasonable
        "equilibration_nvt_ps": 20.0,
        "equilibration_npt_ps": 50.0,
        "energy_minimization_steps": 100
    }
    
    # Clean prev run
    if os.path.exists(config["output_base_dir"]):
        shutil.rmtree(config["output_base_dir"])
        
    engine = MolecularDynamicsEngine(config=config)
    
    # Aspirin
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol_id = "test_aspirin"
    
    print(f"🚀 Running pipeline for {mol_id}...")
    result = engine.run(smiles, molecule_id=mol_id)
    
    # Checks
    if result["status"] != "success":
        print(f"❌ Pipeline Failed: {result.get('message')}")
        sys.exit(1)
        
    print("✅ Pipeline Completed Successfully!")
    
    # Check outputs
    out_dir = result["output_path"]
    expected_files = [
        "trajectory.dcd", "topology.pdb", "state_data.csv", 
        "metadata.json", "simulation_report.json"
    ]
    
    missing = [f for f in expected_files if not os.path.exists(os.path.join(out_dir, f))]
    if missing:
        print(f"❌ Missing output files: {missing}")
        sys.exit(1)
    else:
        print("✅ All output files generated.")
        
    # Check Report Content
    import json
    with open(os.path.join(out_dir, "simulation_report.json")) as f:
        report = json.load(f)
        
    score = report.get("validation", {}).get("stability_score")
    print(f"📊 Stability Score: {score}")
    
    if score is None:
        print("❌ Stability Score missing from report.")
        sys.exit(1)
        
    print("✅ Verification Passed!")

if __name__ == "__main__":
    verify()
