
"""
verify_design_workflow.py — Full verification of the Design & Simulate Workflow with plotting.

1. Simulates Parent (Aspirin).
2. Modifies to Variant (Methyl-Aspirin).
3. Simulates Variant.
4. Generates Comparative Report & Plots.
"""
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from src.md.md_engine import MolecularDynamicsEngine
from src.workflows.design_and_simulate import DesignAndSimulate

def verify():
    print("🧪 Verifying Design & Simulate Workflow...")
    
    # Config: Fast Demo Mode
    config = {
        "mode": "demo",
        "output_base_dir": "data/verify_design",
        "production_ns": 0.01, # 10 ps
        "equilibration_nvt_ps": 1.0, # minimal
        "equilibration_npt_ps": 1.0, # minimal (skips if small)
        "energy_minimization_steps": 100
    }
    
    if os.path.exists(config["output_base_dir"]):
        shutil.rmtree(config["output_base_dir"])
        
    # 1. Run Parent (Aspirin)
    print("\n🔹 Step 1: Simulating Parent (Aspirin)...")
    parent_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    engine = MolecularDynamicsEngine(config=config)
    parent_res = engine.run(parent_smiles, molecule_id="aspirin_parent")
    
    if parent_res["status"] != "success":
        print(f"❌ Parent Simulation Failed: {parent_res['message']}")
        return

    print(f"✅ Parent Score: {parent_res['metrics']['stability_score']:.1f}")

    # 2. Modify & Simulate Variant
    print("\n🔹 Step 2: Modifying -> Methyl-Aspirin...")
    workflow = DesignAndSimulate()
    
    # We override the internal log path for this test to be self-contained if needed,
    # but DesignAndSimulate uses hardcoded path in constructor. That's fine.
    
    variant_res = workflow.modify_and_simulate(
        parent_smiles=parent_smiles,
        modification_type="Add Methyl",
        md_config=config,
        molecule_label="aspirin_variant"
    )
    
    print(f"✅ Variant Generated: {variant_res['modified_smiles']}")
    print(f"✅ Variant Score: {variant_res['md_metrics']['stability_score']:.1f}")
    
    # 3. Compare
    print("\n🔹 Step 3: Comparative Analysis")
    comp = workflow.compare_variants(parent_res["metrics"], variant_res)
    
    print(f"📊 Verdict: {comp['verdict']}")
    print(f"   Stability Change: {comp['stability_change']:+.1f}")
    print(f"   RMSD Change:      {comp['rmsd_change']:+.3f} nm")
    print(f"   SASA Change:      {comp['sasa_change']:+.1f} nm²")
    
    # 4. Generate Plots for Variant
    print("\n🔹 Step 4: Generating Plots...")
    out_dir = variant_res["output_path"]
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, "state_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().replace('"', '') for c in df.columns]
        
        # Energy
        plt.figure(figsize=(10, 5))
        plt.plot(df['Step'], df['Potential Energy (kJ/mole)'], label='Potential Energy')
        plt.title(f"Variant Energy Landscape ({comp['verdict']})")
        plt.xlabel("Step")
        plt.ylabel("Energy (kJ/mol)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, "variant_energy.png"))
        plt.close()
        
        # RMSD (if we had it vs time, but state_data usually just has thermo)
        # We can plot Temperature
        plt.figure(figsize=(10, 5))
        plt.plot(df['Step'], df['Temperature (K)'], color='orange', label='Temperature')
        plt.axhline(300, color='k', linestyle='--')
        plt.title("Variant Temperature Stability")
        plt.ylabel("Temperature (K)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, "variant_temp.png"))
        plt.close()
        
        print(f"✅ Plots saved to: {plot_dir}")
        print("   - variant_energy.png")
        print("   - variant_temp.png")
    else:
        print("❌ Could not find CSV for plotting.")

if __name__ == "__main__":
    verify()
