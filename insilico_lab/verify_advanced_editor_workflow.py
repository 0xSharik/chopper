
"""
verify_advanced_editor_workflow.py — Verify backend logic for Advanced Editor integration.
"""
import os
import shutil
from src.chemistry.atom_tools import get_attachable_atoms
from src.chemistry.reaction_engine import attach_group
from src.workflows.design_and_simulate import DesignAndSimulate

def verify():
    print("🧪 Verifying Advanced Editor Backend Logic...")
    
    # Setup
    parent_smiles = "c1ccccc1" # Benzene
    config = {
        "mode": "demo",
        "output_base_dir": "data/verify_adv_editor",
        "production_ns": 0.002, 
        "equilibration_nvt_ps": 1.0,
        "equilibration_npt_ps": 1.0,
        "energy_minimization_steps": 100
    }
    
    if os.path.exists(config["output_base_dir"]):
        shutil.rmtree(config["output_base_dir"])
        
    # 1. Simulate UI: Analyze Atoms
    print(f"\n🔹 Step 1: Analyzing {parent_smiles}...")
    attachable = get_attachable_atoms(parent_smiles)
    print(f"   Attachable Indices: {attachable}")
    
    if not attachable:
        print("❌ Failed to find attachable atoms")
        return
        
    # 2. Simulate UI: Attach Group (e.g. Hydroxyl to first atom)
    target_idx = attachable[0]
    group = "hydroxyl"
    print(f"\n🔹 Step 2: Attaching '{group}' at index {target_idx}...")
    
    new_smiles = attach_group(parent_smiles, group, target_idx)
    print(f"   Modified SMILES: {new_smiles}")
    
    if not new_smiles:
        print("❌ Attachment failed")
        return
        
    # 3. Simulate Workflow Execution
    print("\n🔹 Step 3: Running Simulation via DesignAndSimulate...")
    designer = DesignAndSimulate()
    
    try:
        result = designer.simulate_modified_smiles(
            parent_smiles=parent_smiles,
            modified_smiles=new_smiles,
            modification_desc=f"Advanced: {group} at {target_idx}",
            md_config=config,
            molecule_label="test_adv_editor"
        )
        
        print(f"✅ Simulation Complete!")
        print(f"   Stability Score: {result['md_metrics']['stability_score']:.1f}")
        print(f"   Output Path: {result['output_path']}")
        
    except Exception as e:
        print(f"❌ Simulation Failed: {e}")

if __name__ == "__main__":
    verify()
