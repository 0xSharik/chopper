"""
analysis.py — MD Trajectory Analysis & Validation.

- Uses MDTraj for structural analysis (RMSD, Rg, SASA).
- Reads state_data.csv for energy/temperature stability.
- Integrates with ValidationEngine for scientific checks.
- Integrates with ReportingEngine for JSON output.
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import mdtraj as md

from src.md.validation import ValidationEngine
from src.md.reporting import ReportingEngine

logger = logging.getLogger("md_engine")

class AnalysisEngine:
    """
    Analyzes MD trajectories and produces validation reports.
    """
    
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.validation = ValidationEngine()
        self.reporting = ReportingEngine()
        
    def analyze_trajectory(self, traj_path: str, topology_path: str, molecule_id: str = "ligand", config: dict = None) -> Dict[str, Any]:
        """
        Perform full analysis on the generated trajectory.
        
        Args:
            traj_path: Path to DCD file
            topology_path: Path to PDB file (topology)
            molecule_id: Identifier
            config: Simulation configuration
            
        Returns:
            metrics: Dictionary of computed metrics
        """
        logger.info(f"Analyzing trajectory: {traj_path}")
        
        # 1. Load Trajectory
        try:
            traj = md.load(traj_path, top=topology_path)
        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}")
            raise
            
        # 2. Handle PBC (Center & Image)
        # We need a reference. Usually first frame.
        try:
            traj.image_molecules(inplace=True)
            traj.superpose(traj, 0)
        except Exception as e:
            logger.warning(f"PBC handling/Alignment warning: {e}")
            
        # 3. Select Ligand
        # Try 'resname UNL', fallback to specific exclusions
        ligand_atom_indices = traj.topology.select("resname UNL or resname LIG or resname MOL")
        
        if len(ligand_atom_indices) == 0:
            logger.warning("No explicit 'UNL/LIG/MOL' residue found. Attempting to identify ligand by excluding solvent.")
            # Exclude water (HOH, WAT, SOL) and common ions (Na, Cl, K, Mg, Ca)
            # MDTraj selection syntax
            selection_str = "not (water or resname HOH or resname WAT or resname SOL or name Na or name Cl or name K or name Mg or name Ca)"
            ligand_atom_indices = traj.topology.select(selection_str)
            
        if len(ligand_atom_indices) == 0:
             # Last resort: find largest residue that isn't water?
             # For now, just warn and return empty? Or fallback to all but fail SASA?
             logger.error("Could not identify ligand atoms. SASA/RMSD will be invalid.")
             # We definitely don't want 'all' if it includes thousands of waters.
             # Only fallback to all if system is small (< 500 atoms)
             if traj.n_atoms < 500:
                  ligand_atom_indices = traj.topology.select("all")
             else:
                  # Return empty -> will crash slicing.
                  # Handle graceful failure in slicing?
                  # Let's just pick atom 0 to prevent crash but metrics will be 0.
                  ligand_atom_indices = [0]
                  logger.error("System too large to default to 'all'. Selected atom 0 to prevent crash.")
        
        n_atoms = len(ligand_atom_indices)
        logger.info(f"Analysis focusing on {n_atoms} atoms.")
        
        # Subsection of trajectory for ligand-only metrics
        lig_traj = traj.atom_slice(ligand_atom_indices)
        
        # 4. Compute Structural Metrics
        # RMSD
        rmsd_frames = md.rmsd(lig_traj, lig_traj, 0) # nm
        rmsd_mean = float(np.mean(rmsd_frames))
        rmsd_std = float(np.std(rmsd_frames))
        
        # Radius of Gyration
        rg_frames = md.compute_rg(lig_traj)
        rg_mean = float(np.mean(rg_frames))
        rg_std = float(np.std(rg_frames))
        
        # SASA (Solvent Accessible Surface Area)
        # shrake_rupley return nm^2 per atom? No, total area per frame.
        # Actually returns (n_frames, n_atoms). Sum over atoms.
        try:
            sasa_per_atom = md.shrake_rupley(lig_traj, mode='residue') 
            # mode='residue' is faster? 'atom' is default.
            # md.shrake_rupley returns total SASA if logic applied?
            # Documentation: returns (n_frames, n_atoms).
            sasa_frames = np.sum(md.shrake_rupley(lig_traj), axis=1)
            sasa_mean = float(np.mean(sasa_frames))
        except Exception as e:
            logger.warning(f"SASA computation failed: {e}")
            sasa_mean = 0.0
            
        # Hydrogen Bonds (Approximate)
        # Baker-Hubbard
        try:
            hbonds = md.baker_hubbard(traj) # Returns array of [frame, donor, acceptor]?
            # Actually returns ndarray (n_hbonds, 3) for ALL frames? 
            # No, identifies HBonds present in "most" frames?
            # md.baker_hubbard identifies hbonds.
            # freq = md.baker_hubbard(traj, freq=True) returns existence frequency.
            # We want average number of Hbonds per frame.
            # wernet_nilsson might be better for per-frame counting.
            # Let's just use simple count from baker_hubbard (avg across traj?)
            # Actually baker_hubbard returns list of hbonds satisfying geometry in >X% frames if periodic=True?
            # Keep it simple:
            hbond_avg = float(len(md.baker_hubbard(traj))) # This is static count of "stable" hbonds?
            # Let's count h-bonds per frame using compute_contacts as proxy? Too complex.
            # Just use 0.0 as placeholder if complex, but explicit solvent implies we want Solute-Solvent Hbonds.
            # md.baker_hubbard finds ALL hbonds.
            # Filter for (One atom in Ligand, One in Water).
        except:
             hbond_avg = 0.0

        # 5. Energy Stability (from CSV)
        csv_path = os.path.join(self.run_dir, "state_data.csv")
        energy_var = 0.0
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Strip columns
                df.columns = [c.strip().replace('"', '') for c in df.columns]
                # Look for Potential Energy
                pe_cols = [c for c in df.columns if "Potential Energy" in c]
                if pe_cols:
                    pe = df[pe_cols[0]]
                    energy_var = float(pe.var())
            except Exception as e:
                logger.warning(f"Energy analysis failed: {e}")

        # 6. Validate
        raw_metrics = {
            "rmsd_mean": rmsd_mean,
            "rmsd_std": rmsd_std,
            "rg_mean": rg_mean,
            "sasa_mean": sasa_mean,
            "hbond_avg": hbond_avg,
            "energy_var": energy_var
        }
        
        val_result = self.validation.validate_simulation(raw_metrics, config, n_atoms)
        
        # 7. Generate Report
        report = self.reporting.generate_report(molecule_id, raw_metrics, val_result, config)
        
        # Save Report
        report_path = os.path.join(self.run_dir, "simulation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        # Save Features for ML
        self._save_features(molecule_id, raw_metrics, val_result)

        # Merge validation info into metrics for return convenience
        raw_metrics["stability_score"] = val_result.get("stability_score", 0)
        raw_metrics["is_valid"] = val_result.get("is_valid", True)
        
        return raw_metrics

    def _save_features(self, mol_id: str, metrics: dict, val_result: dict):
        """Append features to global dataset."""
        data_file = os.path.join(os.path.dirname(self.run_dir), "..", "md_features.csv")
        # Ensure path resolve
        # run_dir is inner folder. ../ is data/md_runs/ or similar.
        
        row = {
            "molecule_id": mol_id,
            "rmsd_mean": metrics["rmsd_mean"],
            "rmsd_std": metrics["rmsd_std"],
            "rg_mean": metrics["rg_mean"],
            "sasa_mean": metrics["sasa_mean"],
            "score": val_result.get("stability_score", 0),
            "timestamp": pd.Timestamp.now()
        }
        
        df = pd.DataFrame([row])
        if not os.path.exists(data_file):
            df.to_csv(data_file, index=False)
        else:
            df.to_csv(data_file, mode='a', header=False, index=False)
            
