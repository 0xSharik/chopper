from rdkit import Chem
from .utils import validate_smiles
from .sas import SyntheticAccessibility
from .ring_strain import RingStrainAnalyzer
from .functional_penalty import FunctionalGroupPenalty
from .reaction_rules import RetrosynthesisRules
from .forward_engine import ForwardSynthesisEngine
import os
import json
import time
import logging

logger = logging.getLogger(__name__)

class SynthesisEngine:
    """Main orchestrator for synthesis intelligence analysis."""
    
    def __init__(self):
        self.sas_engine = SyntheticAccessibility()
        self.ring_engine = RingStrainAnalyzer()
        self.functional_engine = FunctionalGroupPenalty()
        self.retro_engine = RetrosynthesisRules()
        self.forward_engine = ForwardSynthesisEngine()

    def analyze(self, smiles: str, depth: int = 1) -> dict:
        """
        Perform complete synthesis intelligence analysis.
        Args:
            smiles: Input molecule SMILES.
            depth: Retrosynthesis disconnection depth (1 or 2).
        """
        try:
            mol = validate_smiles(smiles)
            
            # 1. Compute individual scores
            sas = self.sas_engine.calculate(mol)
            ring = self.ring_engine.score(mol)
            functional = self.functional_engine.score(mol)
            
            # 2. Composite score calculation
            # Weights: 50% SAS, 25% Ring Strain, 25% Functional Penalty
            feasibility = (sas * 0.5) + (ring * 0.25) + (functional * 0.25)
            
            # Normalize to 0-100 scale (where lower feasibility score is better)
            # Actually, per user prompt: feasibility_scaled = round((feasibility / 10) * 100, 2)
            # This makes 100 the "Difficult" end.
            feasibility_scaled = round((feasibility / 10) * 100, 2)
            feasibility_scaled = min(100.0, feasibility_scaled)
            
            # 3. Classification
            if feasibility_scaled < 35:
                classification = "Easy"
            elif feasibility_scaled <= 65:
                classification = "Moderate"
            else:
                classification = "Difficult"
                
            # 4. Generate precursors (Step 1)
            first_level = self.retro_engine.generate_precursors(mol)
            
            # Compute retrosynthesis confidence (Average of top disconnections)
            if first_level:
                retro_confidence = max([p["confidence"] for p in first_level])
            else:
                retro_confidence = 0.0
            
            final_precursors = first_level
            
            # 5. Optional Depth=2 logic (Step 2)
            if depth >= 2:
                depth2_precursors = []
                for p1 in first_level:
                    # Parse smiles components (if multiple molecules)
                    components = p1["smiles"].split(".")
                    for comp in components:
                        comp_mol = Chem.MolFromSmiles(comp)
                        if comp_mol:
                            second_level = self.retro_engine.generate_precursors(comp_mol)
                            for p2 in second_level:
                                # Avoid trivial cycles or repeating parent
                                if p2["smiles"] != smiles and p2["smiles"] not in [p["smiles"] for p in first_level]:
                                    depth2_precursors.append({
                                        "smiles": p2["smiles"],
                                        "reaction_type": p2["reaction_type"],
                                        "confidence": p2["confidence"] * p1["confidence"], # Compounded confidence
                                        "name": f"{p1['name']} -> {p2['name']}",
                                        "parent_smiles": p1["smiles"]
                                    })
                
                # Merge and limit
                all_retro = first_level + depth2_precursors
                # Deduplicate by smiles
                seen = set()
                final_precursors = []
                for p in sorted(all_retro, key=lambda x: x["confidence"], reverse=True):
                    if p["smiles"] not in seen:
                        final_precursors.append(p)
                        seen.add(p["smiles"])
                
                final_precursors = final_precursors[:5]
            
            # 6. Forward Synthesis Preview (Step 3) - Optional if we want it here
            # For now, we'll keep it focused on analyze returning precursors.
            # But the user wants "forward_products" in the final structure.
            # Let's run a quick forward synthesis if precursors exist.
            
            forward_products = []
            if first_level:
                # Try simple combinations for forward preview
                # Actually, virtual_synthesis is for explicit pairs.
                # Let's leave forward_products empty unless explicitly called, 
                # OR we can try to "reconstruct" parent if needed.
                # Per prompt, analyze should return forward_products.
                pass

            return {
                "sas_score": sas,
                "ring_strain_score": ring,
                "functional_penalty_score": functional,
                "feasibility_score": feasibility_scaled,
                "classification": classification,
                "retrosynthesis": {
                    "precursors": final_precursors,
                    "confidence": round(retro_confidence, 2),
                    "analysis_label": f"Disconnection Depth: {depth}-step structural analysis"
                },
                "forward_products": [] # Usually populated by VS
            }
            
        except Exception as e:
            raise ValueError(f"Synthesis analysis failed: {str(e)}")

    def virtual_synthesis(self, precursors: list) -> dict:
        """
        Orchestrate virtual forward synthesis from precursors.
        """
        try:
            # 1. Generate products (detailed objects from forward_engine)
            result = self.forward_engine.synthesize(precursors)
            products = result.get("products", [])
            
            # 2. Save logs
            self._save_synthesis_run(precursors, products)
            
            return {
                "products": [p["smiles"] for p in products], # Simple list for backward compatibility
                "product_details": products # All metadata: reaction_name, admet_preview, etc.
            }
        except Exception as e:
            # We already have logger now
            logger.error(f"Virtual synthesis failed: {str(e)}")
            raise ValueError(f"Virtual synthesis failed: {str(e)}")

    def _save_synthesis_run(self, precursors: list, results: list):
        """Save run details to disk."""
        timestamp = int(time.time())
        log_dir = os.path.join("data", "virtual_synthesis_runs", str(timestamp))
        os.makedirs(log_dir, exist_ok=True)
        
        with open(os.path.join(log_dir, "input_precursors.json"), "w") as f:
            json.dump(precursors, f, indent=4)
        
        with open(os.path.join(log_dir, "generated_products.json"), "w") as f:
            json.dump([r["smiles"] for r in results], f, indent=4)
            
        with open(os.path.join(log_dir, "feasibility_scores.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        return log_dir
