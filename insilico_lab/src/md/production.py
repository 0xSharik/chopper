"""
production.py — MD production run.

Clears equilibration reporters, attaches DCD trajectory reporter and a
StateDataReporter, then runs the full production simulation.

Output files (in output_dir):
  - trajectory.dcd       (DCD format, frame every output_frame_ps)
  - state_data.csv       (Step, Potential Energy, Temperature, Volume)
  - final_state.xml      (OpenMM checkpoint)
  - metadata.json        (config snapshot, version info, timing, box/atom info)
"""

import os
import json
import logging
import datetime
from typing import Tuple

import openmm
import openmm.app as app
import openmm.unit as unit

logger = logging.getLogger("md_engine")

class ProductionEngine:
    """
    Handles the production MD run and output generation.
    """

    def __init__(self, simulation: app.Simulation, config: dict, output_dir: str):
        self.simulation = simulation
        self.config = config
        self.output_dir = output_dir

    def run_production(self) -> str:
        """
        Run the production MD simulation and persist all output files.

        Returns:
            trajectory_path: absolute path to the generated DCD file.
        """
        dt_fs        = self.config["timestep_fs"]
        prod_ns      = self.config["production_ns"]
        frame_ps     = self.config["output_frame_ps"]
        seed         = self.config["seed"]

        total_steps  = int(prod_ns * 1e6 / dt_fs)          # ns → fs → steps
        frame_steps  = int(frame_ps * 1000 / dt_fs)        # ps → fs → steps
        n_frames     = total_steps // frame_steps

        logger.info("=== Production Run ===")
        logger.info(f"  Duration:       {prod_ns} ns  ({total_steps:,} steps)")
        logger.info(f"  Frame interval: {frame_ps} ps  ({frame_steps} steps)")
        logger.info(f"  Expected frames: {n_frames}")
        logger.info(f"  Output dir:     {self.output_dir}")

        # Clear equilibration reporters
        self.simulation.reporters.clear()

        # Output file paths
        traj_path   = os.path.join(self.output_dir, "trajectory.dcd")
        state_path  = os.path.join(self.output_dir, "state_data.csv")
        xml_path    = os.path.join(self.output_dir, "final_state.xml")
        meta_path   = os.path.join(self.output_dir, "metadata.json")

        # DCD trajectory reporter
        self.simulation.reporters.append(
            app.DCDReporter(traj_path, frame_steps)
        )

        # State data reporter (CSV)
        self.simulation.reporters.append(
            app.StateDataReporter(
                state_path,
                frame_steps,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                progress=True,
                remainingTime=True,
                speed=True,
                totalSteps=total_steps,
                separator=",",
            )
        )

        # Run production
        start_time = datetime.datetime.now()
        logger.info(f"  Production started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.simulation.step(total_steps)

        end_time = datetime.datetime.now()
        elapsed  = (end_time - start_time).total_seconds()
        logger.info(f"  Production completed in {elapsed:.1f} s")
        logger.info(f"  → Trajectory: {traj_path}")

        # Save final state XML
        self.simulation.saveState(xml_path)
        logger.info(f"  → Final state: {xml_path}")

        # Save PDB Topology for analysis convenience (if not already saved by SystemBuilder?)
        # AnalysisEngine needs a PDB. SystemBuilder returns Topology.
        # It's good practice to save the topology PDB here or earlier.
        # SystemBuilder doesn't save PDB explicitly to run_dir (it might, but let's ensure it).
        # We can extract topology from simulation and save it.
        pdb_path = os.path.join(self.output_dir, "topology.pdb")
        with open(pdb_path, 'w') as f:
            app.PDBFile.writeFile(self.simulation.topology, self.simulation.context.getState(getPositions=True).getPositions(), f)

        # Write metadata.json
        state = self.simulation.context.getState(getPositions=True)
        box_vectors = state.getPeriodicBoxVectors()
        n_atoms = self.simulation.topology.getNumAtoms()
        
        self._write_metadata(
            meta_path=meta_path,
            n_atoms=n_atoms,
            box_vectors=box_vectors,
            total_steps=total_steps,
            n_frames=n_frames,
            start_time=start_time,
            end_time=end_time,
            elapsed_s=elapsed,
        )
        logger.info(f"  → Metadata:    {meta_path}")

        # Clear reporters to release file locks (Critical for Windows)
        self.simulation.reporters.clear()

        return traj_path

    def _write_metadata(
        self,
        meta_path: str,
        n_atoms: int,
        box_vectors,
        total_steps: int,
        n_frames: int,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        elapsed_s: float,
    ) -> None:
        """
        Write a JSON metadata file capturing all reproducibility information.
        """
        import openmm as _omm
        import openmm.unit as unit

        # Serialize box vectors (list of 3-tuples in nm)
        def _vec_nm(v):
            return [round(float(v[i][i].value_in_unit(unit.nanometers)), 4) for i in range(3)]

        try:
            box_nm = _vec_nm(box_vectors)
            volume_nm3 = box_vectors[0][0].value_in_unit(unit.nanometers) * \
                         box_vectors[1][1].value_in_unit(unit.nanometers) * \
                         box_vectors[2][2].value_in_unit(unit.nanometers)
        except Exception:
            box_nm = None
            volume_nm3 = None
            
        # Calculate Density
        density_val = 0.0
        if volume_nm3:
             total_mass_amu = sum([self.simulation.system.getParticleMass(i).value_in_unit(unit.dalton) 
                                  for i in range(self.simulation.system.getNumParticles())])
             density_val = (total_mass_amu / 6.02214076e23) / (volume_nm3 * 1.0e-21)

        metadata = {
            "molecule_id":      "unknown", 
            "n_atoms":          n_atoms,
            "box_size_nm":      box_nm,
            "density_g_cm3":    round(density_val, 3),
            "padding_nm":       self.config.get("padding_nm"),
            "cutoff_nm":        0.9,
            "pme":              True,
            "barostat_frequency": 100,
            "total_steps":      total_steps,
            "n_frames":         n_frames,
            "simulation_time_ns": self.config["production_ns"],
            "config_used":      self.config,
            "openmm_version":   _omm.__version__,
            "seed":             self.config["seed"],
            "simulation_start": start_time.isoformat(),
            "simulation_end":   end_time.isoformat(),
            "elapsed_seconds":  round(elapsed_s, 2),
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
