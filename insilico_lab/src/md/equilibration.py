"""
equilibration.py — Staged MD equilibration protocol.

Stages:
  A) Energy minimization: 5000 steps, tolerance=10 kJ/mol/nm
  B) NVT equilibration: Langevin thermostat, 300 K, 50 ps, 2 fs step
  C) NPT equilibration: + MonteCarloBarostat, 1 atm, 200 ps

For each stage, energy / temperature / pressure are logged to CSV files
in the run output directory.
"""

import os
import logging
from typing import Tuple, Optional

import openmm
import openmm.app as app
import openmm.unit as unit

logger = logging.getLogger("md_engine")

class EquilibrationEngine:
    """
    Handles the equilibration stages (Minimization -> NVT -> NPT).
    """
    
    def __init__(self, system: openmm.System, topology: app.Topology, positions: list, config: dict, output_dir: str):
        self.system = system
        self.topology = topology
        self.positions = positions
        self.config = config
        self.output_dir = output_dir
        
    def run_equilibration(self) -> app.Simulation:
        """
        Execute the equilibration protocol.
        
        Returns:
            openmm.app.Simulation — equilibrated, ready for production.
        """
        # Determine Platform
        # HEURISTIC: Try CUDA, then OpenCL, then CPU
        platform = None
        props = {}
        
        valid_platforms = [p.getName() for p in [openmm.Platform.getPlatform(i) for i in range(openmm.Platform.getNumPlatforms())]]
        logger.info(f"Available Platforms: {valid_platforms}")
        
        # User preference
        for pref in self.config.get("platform_preference", ["CUDA", "OpenCL", "CPU"]):
            if pref in valid_platforms:
                try:
                    platform = openmm.Platform.getPlatformByName(pref)
                    if pref in ["CUDA", "OpenCL"]:
                        props = {"Precision": "mixed"}
                    logger.info(f"Using Platform: {pref}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to initialize {pref}: {e}")
        
        if platform is None:
             # Fallback
             platform = openmm.Platform.getPlatform(0)
             logger.warning(f"Using default platform: {platform.getName()}")

        return self._run_protocol(platform, props)

    def _run_protocol(self, platform, platform_props):
        T          = self.config["temperature"] * unit.kelvin
        dt         = self.config["timestep_fs"] * unit.femtoseconds
        friction   = self.config["friction_coeff"] / unit.picoseconds
        seed       = self.config["seed"]
        tolerance  = self.config["energy_tolerance_kj"] * unit.kilojoules_per_mole / unit.nanometers

        # ------------------------------------------------------------------ #
        # Stage A: Energy Minimization
        # ------------------------------------------------------------------ #
        logger.info("=== Stage A: Energy Minimization ===")
        # Create Integrator for minimization (Langevin is fine, or Verlet)
        integrator_min = openmm.LangevinMiddleIntegrator(T, friction, dt)
        integrator_min.setRandomNumberSeed(seed)

        simulation = app.Simulation(
            self.topology, self.system, integrator_min, platform, platform_props
        )
        simulation.context.setPositions(self.positions)

        # Report initial PE
        state_pre = simulation.context.getState(getEnergy=True)
        e_pre = state_pre.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        logger.info(f"  Pre-minimization PE: {e_pre:.2f} kJ/mol")

        simulation.minimizeEnergy(
            tolerance=tolerance,
            maxIterations=self.config["energy_minimization_steps"],
        )

        state_post = simulation.context.getState(getEnergy=True, getPositions=True)
        e_post = state_post.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        logger.info(f"  Post-minimization PE: {e_post:.2f} kJ/mol")

        minimized_positions = state_post.getPositions()

        # ------------------------------------------------------------------ #
        # Stage B: NVT Equilibration
        # ------------------------------------------------------------------ #
        nvt_ps    = self.config["equilibration_nvt_ps"]
        nvt_steps = int(nvt_ps * 1000 / self.config["timestep_fs"])

        logger.info(f"=== Stage B: NVT Equilibration ({nvt_ps} ps) ===")

        # New Integrator for NVT
        integrator_nvt = openmm.LangevinMiddleIntegrator(T, friction, dt)
        integrator_nvt.setRandomNumberSeed(seed)
        
        # New Simulation for NVT (clean state)
        sim_nvt = app.Simulation(
            self.topology, self.system, integrator_nvt, platform, platform_props
        )
        sim_nvt.context.setPositions(minimized_positions)
        sim_nvt.context.setVelocitiesToTemperature(T, seed)

        nvt_log = os.path.join(self.output_dir, "nvt_equilibration.csv")
        report_interval = max(1, nvt_steps // 100)
        sim_nvt.reporters.append(
             app.StateDataReporter(nvt_log, report_interval, step=True, potentialEnergy=True, 
                                   temperature=True, separator=",")
        )

        sim_nvt.step(nvt_steps)
        
        state_nvt = sim_nvt.context.getState(getPositions=True, getVelocities=True)
        logger.info("  NVT done.")

        # ------------------------------------------------------------------ #
        # Stage C: NPT Equilibration
        # ------------------------------------------------------------------ #
        npt_ps    = self.config["equilibration_npt_ps"]
        npt_steps = int(npt_ps * 1000 / self.config["timestep_fs"])
        P         = self.config["pressure"] * unit.atmospheres
        
        # Guard: Demo Mode Safety Skip (Check FIRST)
        n_water = len([r for r in self.topology.residues() if r.name == 'HOH'])
        mode = self.config.get("mode", "research")
        
        if mode == "demo" and n_water < 800:
            logger.warning(f"Demo mode: Skipping NPT due to small system ({n_water} waters).")
            return sim_nvt

        # Guard: NPT Duration (Check SECOND)
        if npt_ps < 20.0:
            raise ValueError(f"NPT duration {npt_ps} ps is too short for stable equilibration. Minimum 20 ps required.")

        logger.info(f"=== Stage C: NPT Equilibration ({npt_ps} ps) ===")

        # Guard: Box Size vs Cutoff
        # Cutoff is 0.9 nm (enforced in SystemBuilder)
        cutoff_nm = 0.9
        box_vectors = sim_nvt.context.getState().getPeriodicBoxVectors()
        lx = box_vectors[0][0].value_in_unit(unit.nanometers)
        ly = box_vectors[1][1].value_in_unit(unit.nanometers)
        lz = box_vectors[2][2].value_in_unit(unit.nanometers)
        min_box_dim = min(lx, ly, lz)
        
        if min_box_dim < 2 * cutoff_nm:
            raise RuntimeError(f"Box dimension ({min_box_dim:.2f} nm) smaller than 2x cutoff ({2*cutoff_nm} nm). Unsafe to proceed with NPT.")

        # Add Barostat
        # Check if barostat already exists
        has_barostat = any(isinstance(f, openmm.MonteCarloBarostat) for f in self.system.getForces())
        if not has_barostat:
            # Update: Frequency 100 for stability
            barostat = openmm.MonteCarloBarostat(P, T, 100)
            barostat.setRandomNumberSeed(seed)
            self.system.addForce(barostat)

        integrator_npt = openmm.LangevinMiddleIntegrator(T, friction, dt)
        integrator_npt.setRandomNumberSeed(seed)
        
        sim_npt = app.Simulation(
            self.topology, self.system, integrator_npt, platform, platform_props
        )
        sim_npt.context.setPositions(state_nvt.getPositions())
        sim_npt.context.setVelocities(state_nvt.getVelocities())
        
        npt_log = os.path.join(self.output_dir, "npt_equilibration.csv")
        report_interval_npt = max(1, npt_steps // 100)
        sim_npt.reporters.append(
             app.StateDataReporter(npt_log, report_interval_npt, step=True, potentialEnergy=True, 
                                   temperature=True, volume=True, density=True, separator=",")
        )

        sim_npt.step(npt_steps)
        logger.info("  NPT done.")
        
        # Patched: Density Monitoring
        state = sim_npt.context.getState(getPositions=True)
        box = state.getPeriodicBoxVectors()
        volume = box[0][0] * box[1][1] * box[2][2]
        volume_nm3 = volume.value_in_unit(unit.nanometers**3)
        
        total_mass = sum([openmm.app.Element.getBySymbol(a.element.symbol).mass.value_in_unit(unit.dalton) 
                          for a in self.topology.atoms() if a.element])
        # Note: topology.atoms() + element mass is safer than system.getParticleMass if virtual sites exist? 
        # But system.getParticleMass is authoritative for simulation physics.
        # Let's use system
        total_mass_amu = sum([self.system.getParticleMass(i).value_in_unit(unit.dalton) 
                              for i in range(self.system.getNumParticles())])
        
        density_g_cm3 = (total_mass_amu / 6.022e23) / (volume_nm3 * 1e-21) # nm^3 to cm^3 is 1e-21 ? 
        # 1 nm = 1e-7 cm. (1e-7)^3 = 1e-21. Correct.
        # Wait, prompt said 1e-24? 
        # Prompt: "density = (mass_amu / 6.022e23) / (volume_nm3 * 1e-24) # g/cm3"
        # Let's check: 1 nm = 10^-9 m = 10^-7 cm.
        # 1 nm^3 = (10^-7)^3 cm^3 = 10^-21 cm^3.
        # User prompt might have typo or I am misremembering unit conversion.
        # 1e-24 would be for A^3? 1 A = 10^-8 cm. 1 A^3 = 10^-24 cm^3.
        # Volume is in nm^3. So 1e-21 is correct for nm^3. 
        # I will use 1e-21 if volume is in nm^3.
        # Ah, maybe prompt volume was in A^3? No, "volume.value_in_unit(nanometer**3)".
        # I will use 1e-21.
        
        density = (total_mass_amu / 6.02214076e23) / (volume_nm3 * 1.0e-21)
        
        logger.info(f"System density: {density:.3f} g/cm3")
        if density < 0.8 or density > 1.2:
             logger.warning(f"Density {density:.3f} g/cm3 out of expected range (0.8–1.2 g/cm3)")

        return sim_npt
