
"""
comparison_engine.py — Logic to compare MD metrics between Parent and Variant.
"""
import logging

logger = logging.getLogger("design_workflow")

class MoleculeComparison:
    """
    Compares two sets of MD metrics and determines a stability verdict.
    """
    
    @staticmethod
    def compare(parent_metrics: dict, variant_metrics: dict) -> dict:
        """
        Compare metrics and return delta analysis.
        
        Args:
            parent_metrics: dict from MD Engine (rmsd_mean, etc.)
            variant_metrics: dict from MD Engine
            
        Returns:
            dict with deltas and verdict.
        """
        # Defaults if metrics missing
        p_rmsd = parent_metrics.get("rmsd_mean", 0.0)
        v_rmsd = variant_metrics.get("rmsd_mean", 0.0)
        
        p_rg = parent_metrics.get("rg_mean", 0.0)
        v_rg = variant_metrics.get("rg_mean", 0.0)
        
        p_sasa = parent_metrics.get("sasa_mean", 0.0)
        v_sasa = variant_metrics.get("sasa_mean", 0.0)
        
        p_score = parent_metrics.get("stability_score", 0.0)
        v_score = variant_metrics.get("stability_score", 0.0)
        
        # Calculate Deltas (Variant - Parent)
        delta_rmsd = v_rmsd - p_rmsd
        delta_rg = v_rg - p_rg
        delta_sasa = v_sasa - p_sasa
        delta_score = v_score - p_score
        
        # Verdict Logic
        verdict = "Neutral"
        
        # Significant score change check
        if delta_score > 5.0:
            verdict = "More Stable (Score)"
        elif delta_score < -5.0:
            verdict = "Less Stable (Score)"
        else:
            # Secondary checks if score is similar
            if delta_rmsd < -0.05:
                verdict = "More Rigid (RMSD)"
            elif delta_rmsd > 0.05:
                verdict = "Less Rigid (RMSD)"
            elif delta_sasa < -1.0:
                 verdict = "More Compact (SASA)"
        
        return {
            "rmsd_change": round(delta_rmsd, 3),
            "rg_change": round(delta_rg, 3),
            "sasa_change": round(delta_sasa, 2),
            "stability_change": round(delta_score, 1),
            "original_score": round(p_score, 1),
            "variant_score": round(v_score, 1),
            "verdict": verdict
        }
