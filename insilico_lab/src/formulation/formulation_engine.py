import os
import time
import logging
from typing import Dict, Any, List

from src.md.config import DEFAULT_MD_CONFIG
from src.md.equilibration import EquilibrationEngine
from src.md.production import ProductionEngine
from src.md.analysis import AnalysisEngine

from .mixture_parser import MixtureParser
from .composition_engine import CompositionEngine
from .multi_system_builder import MultiMoleculeSystemBuilder
from .interaction_analysis import InteractionAnalyzer

logger = logging.getLogger("md_engine")

class FormulationEngine:
    """Orchestrates multi-molecule formulation simulation and analysis."""
    
    def __init__(self, config: Dict = None):
        self.config = DEFAULT_MD_CONFIG.copy()
        if config:
            self.config.update(config)
            
        self.parser = MixtureParser()
        self.composition_engine = CompositionEngine()
        self.system_builder = MultiMoleculeSystemBuilder(self.config)
        self.interaction_analyzer = InteractionAnalyzer()

    def run(self, mixture_input: Any, formulation_id: str = "formulation_1") -> Dict:
        """
        Run the formulation simulation pipeline.
        """
        start_time = time.time()
        logger.info(f"Starting Formulation Pipeline for '{formulation_id}'")
        logger.info(f"   Platform Preference: {self.config.get('platform_preference')}")
        
        # 1. Parse and Normalize
        mixture_raw = self.parser.parse(mixture_input)
        mixture = self.composition_engine.normalize(mixture_raw, max_total=self.config.get("max_solutes", 50))
        summary = self.composition_engine.get_summary(mixture)
        
        run_dir = os.path.join(self.config["output_base_dir"], formulation_id)
        os.makedirs(run_dir, exist_ok=True)
        
        result = {
            "formulation_id": formulation_id,
            "composition": mixture,
            "summary": summary,
            "status": "pending",
            "output_path": run_dir
        }
        
        try:
            # 2. Build Multi-Solute System
            logger.info("Step 1: Multi-Solute System Building")
            topology, system, positions = self.system_builder.build(mixture, formulation_id)
            
            # 3. Equilibration
            logger.info("Step 2: Equilibration")
            equilibrator = EquilibrationEngine(system, topology, positions, self.config, run_dir)
            simulation = equilibrator.run_equilibration()
            
            # 4. Production
            logger.info("Step 3: Production MD")
            producer = ProductionEngine(simulation, self.config, run_dir)
            traj_path = producer.run_production()
            
            # 5. Analysis
            logger.info("Step 4: Interaction Analysis")
            pdb_path = os.path.join(run_dir, "topology.pdb")
            # Save PDB for analysis
            import openmm.app as app
            with open(pdb_path, 'w') as f:
                app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
            
            interaction_metrics = self.interaction_analyzer.analyze(traj_path, pdb_path)
            
            # 6. Stability Scoring
            stability_score = self._compute_stability_score(interaction_metrics)
            
            result.update({
                "interaction_metrics": interaction_metrics,
                "stability_score": stability_score,
                "classification": self._classify_stability(stability_score, interaction_metrics),
                "status": "success"
            })
            
        except Exception as e:
            logger.error(f"Formulation Pipeline Failed: {e}", exc_info=True)
            result["status"] = "failed"
            result["message"] = str(e)
            
        result["elapsed_seconds"] = time.time() - start_time
        return result

    def _compute_stability_score(self, metrics: Dict) -> float:
        """Compute a 0-100 stability score."""
        agg_index = metrics.get("aggregation_index", 0.0)
        # Higher aggregation index reduces stability score
        score = 100.0 * (1.0 - agg_index)
        return round(max(0.0, score), 2)

    def _classify_stability(self, score: float, metrics: Dict) -> str:
        agg_index = metrics.get("aggregation_index", 0.0)
        if agg_index > 0.7:
            return "Aggregating"
        elif agg_index > 0.3:
            return "Incompatible"
        else:
            return "Stable"
