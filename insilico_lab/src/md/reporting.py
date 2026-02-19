"""
reporting.py — MD Results Formatting & Reporting.

Converts raw validation data into user-friendly JSON/Dict structures for the GUI.
Ensures consistency in output format regardless of run mode (Demo/Research).
"""

import json
import os
from src.md.config import DEFAULT_MD_CONFIG

class ReportingEngine:
    """
    Generates structured reports for MD simulations.
    """
    
    @staticmethod
    def generate_report(molecule_id: str, metrics: dict, validation: dict, config: dict) -> dict:
        """
        Create a full JSON-serializable report for the run.
        
        Args:
            molecule_id: Unique identifier
            metrics: Analysis results (RMSD, etc.)
            validation: Validation engine output (is_valid, warnings, score)
            config: Simulation configuration
            
        Returns:
            dict with structured report
        """
        report = {
            "molecule_id": molecule_id,
            "status": "success" if validation.get("is_valid", True) else "failed",
            "mode": config.get("mode", "unknown"),
            
            # Key Analysis Metrics
            "metrics": {
                "rmsd_mean": metrics.get("rmsd_mean", 0.0),
                "rmsd_std": metrics.get("rmsd_std", 0.0),
                "rg_mean": metrics.get("rg_mean", 0.0),
                "sasa_mean": metrics.get("sasa_mean", 0.0), # nm^2
                "hbond_avg": metrics.get("hbond_avg", 0.0),
                "energy_variance": metrics.get("energy_var", 0.0),
            },
            
            # Validation & Stability
            "validation": {
                "is_valid": validation.get("is_valid", True),
                "warnings": validation.get("warnings", []),
                "stability_score": validation.get("stability_score", 0.0),
                "integrity_check": validation.get("integrity_check", {})
            },
            
            # Configuration Details (for reproducibility)
            "config": {
                "temperature": config.get("temperature"),
                "pressure": config.get("pressure"),
                "production_ns": config.get("production_ns"),
                "water_model": config.get("water_model"),
                "seed": config.get("seed")
            }
        }
        
        return report

    @staticmethod
    def save_report(report: dict, output_dir: str):
        """Save report to JSON file."""
        path = os.path.join(output_dir, "simulation_report.json")
        try:
            with open(path, "w") as f:
                json.dump(report, f, indent=4)
        except Exception as e:
            print(f"Error saving report: {e}")
