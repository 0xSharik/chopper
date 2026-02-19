
"""
design_and_simulate.py — Orchestrates the Modify -> Validate -> Simulate workflow.
"""
import logging
import os
import pandas as pd
import datetime

from src.md.md_engine import MolecularDynamicsEngine
from src.chemistry import modifier_engine
from src.workflows.validation import validate_smiles, check_modification_reasonable
from src.workflows.comparison_engine import MoleculeComparison

logger = logging.getLogger("design_workflow")

class DesignAndSimulate:
    """
    Workflow manager for iterative drug design and simulation.
    """
    
    def __init__(self):
        # Result log file
        self.result_log_path = os.path.join("data", "md_variant_results.csv")
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.result_log_path):
            df = pd.DataFrame(columns=[
                "parent_smiles", "modified_smiles", "modification", 
                "rmsd", "rg", "sasa", "hbond", "stability_score", "timestamp"
            ])
            df.to_csv(self.result_log_path, index=False)

    def modify_and_simulate(
        self,
        parent_smiles: str,
        modification_type: str,
        md_config: dict,
        molecule_label: str = "variant"
    ) -> dict:
        """
        Executes the full modification and simulation pipeline.
        
        Args:
            parent_smiles: config str
            modification_type: str (methyl, fluorine, etc.)
            md_config: dict for MD Engine
            molecule_label: str
            
        Returns:
            dict with full results and comparison.
        """
        logger.info(f"Starting Design Workflow: {modification_type} on {molecule_label}")
        
        # 1. Generate New SMILES
        # Map modification key to modifier_engine function
        modifier_map = {
            "Add Methyl": modifier_engine.add_methyl,
            "Add Fluorine": modifier_engine.add_fluorine,
            "Add Hydroxyl": modifier_engine.add_hydroxyl,
            "Extend Chain": modifier_engine.extend_alkyl_chain,
            "Add Chlorine": modifier_engine.add_chlorine
        }
        
        modifier_func = modifier_map.get(modification_type)
        if not modifier_func:
            raise ValueError(f"Unknown modification type: {modification_type}")
            
        new_smiles = modifier_func(parent_smiles)
        
        if not new_smiles or new_smiles == parent_smiles:
             raise ValueError("Modification failed to generate a new valid structure.")
             
        logger.info(f"Generated Variant: {new_smiles}")
        
        # 2. Validate New Structure
        try:
            validate_smiles(new_smiles)
            check_modification_reasonable(parent_smiles, new_smiles)
        except ValueError as e:
            raise ValueError(f"Validation Violation: {e}")
            
        # 3. Rebuild & Simulate (Fresh Engine)
        # IMPORTANT: Each modified molecule is treated as a completely new physical system.
        # We instantiate a new engine to ensure no parameter leakage.
        
        # Enforce NVT minimums for safety if needed, rely on config passed in.
        # But we must respect the "Demo mode safe skip" logic in the engine.
        
        engine = MolecularDynamicsEngine(config=md_config)
        
        # Label the run
        run_id = f"{molecule_label}_{modification_type.replace(' ', '_').lower()}"
        
        logger.info(f"Launching MD for {run_id}...")
        results = engine.run(new_smiles, molecule_id=run_id)
        
        if results["status"] != "success":
             raise RuntimeError(f"MD Simulation Failed: {results.get('message')}")
             
        metrics = results["metrics"]
        
        # 4. Log Results
        self._log_result(parent_smiles, new_smiles, modification_type, metrics)
        
        return {
            "parent_smiles": parent_smiles,
            "modified_smiles": new_smiles,
            "modification": modification_type,
            "md_metrics": metrics,
            "output_path": results["output_path"]
        }

    def simulate_modified_smiles(
        self,
        parent_smiles: str,
        modified_smiles: str,
        modification_desc: str,
        md_config: dict,
        molecule_label: str = "variant"
    ) -> dict:
        """
        Simulate a pre-modified molecule (e.g., from Advanced Editor).
        
        Args:
            parent_smiles: Original SMILES
            modified_smiles: New SMILES
            modification_desc: Description (e.g. "Methyl at atom 5")
            md_config: Simulation config
            molecule_label: Label
        
        Returns:
            dict result
        """
        logger.info(f"Starting Advanced Design Simulation: {modification_desc}")
        
        # 1. Validate
        if not modified_smiles or modified_smiles == parent_smiles:
             raise ValueError("Modified SMILES is invalid or identical to parent.")
             
        try:
            validate_smiles(modified_smiles)
            check_modification_reasonable(parent_smiles, modified_smiles)
        except ValueError as e:
            raise ValueError(f"Validation Violation: {e}")
            
        # 2. Rebuild & Simulate
        engine = MolecularDynamicsEngine(config=md_config)
        
        # Label
        safe_desc = "".join([c if c.isalnum() else "_" for c in modification_desc]).lower()[:20]
        run_id = f"{molecule_label}_adv_{safe_desc}"
        
        logger.info(f"Launching MD for {run_id}...")
        results = engine.run(modified_smiles, molecule_id=run_id)
        
        if results["status"] != "success":
             raise RuntimeError(f"MD Simulation Failed: {results.get('message')}")
             
        metrics = results["metrics"]
        
        # 3. Log
        self._log_result(parent_smiles, modified_smiles, modification_desc, metrics)
        
        return {
            "parent_smiles": parent_smiles,
            "modified_smiles": modified_smiles,
            "modification": modification_desc,
            "md_metrics": metrics,
            "output_path": results["output_path"]
        }
        
    def compare_variants(self, parent_metrics: dict, variant_results: dict):
        """Helper to run comparison logic."""
        return MoleculeComparison.compare(parent_metrics, variant_results["md_metrics"])

    def _log_result(self, parent, variant, mod, metrics):
        """Append to CSV log."""
        try:
            entry = {
                "parent_smiles": parent,
                "modified_smiles": variant,
                "modification": mod,
                "rmsd": metrics.get("rmsd_mean"),
                "rg": metrics.get("rg_mean"),
                "sasa": metrics.get("sasa_mean"),
                "hbond": metrics.get("hbond_avg"),
                "stability_score": metrics.get("stability_score"),
                "timestamp": datetime.datetime.now().isoformat()
            }
            df = pd.DataFrame([entry])
            df.to_csv(self.result_log_path, mode='a', header=False, index=False)
        except Exception as e:
            logger.error(f"Failed to log results: {e}")
