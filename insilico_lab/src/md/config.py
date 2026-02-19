"""
config.py — MD simulation configuration defaults.

Research-grade configuration with support for Demo mode and Full Production mode.
All physical parameters follow OpenMM conventions (ps, nm, kJ/mol, atm, K).
"""

DEFAULT_MD_CONFIG = {
    # ------------------------------------------------------------------
    # Simulation Protocols
    # ------------------------------------------------------------------
    "mode": "demo",  # 'demo' (fast, ~0.2ns) or 'research' (full, ~5ns+)

    # ------------------------------------------------------------------
    # Thermodynamics
    # ------------------------------------------------------------------
    "temperature": 300.0,         # Kelvin
    "pressure": 1.0,              # atm (for NPT barostat)

    # ------------------------------------------------------------------
    # Solvent Box
    # ------------------------------------------------------------------
    "padding_nm": 1.3,            # nm (Increased for NPT stability)
    "water_model": "tip3p",       # solvent model identifier
    "ionic_strength": 0.0,        # Molar
    "neutralize": True,           # Add Na+/Cl- to neutralize system

    # ------------------------------------------------------------------
    # Integrator
    # ------------------------------------------------------------------
    "timestep_fs": 2.0,           # femtoseconds per MD step
    "friction_coeff": 1.0,        # ps^-1 (Langevin thermostat friction)

    # ------------------------------------------------------------------
    # Equilibration Settings (Base values - scaled in demo mode)
    # ------------------------------------------------------------------
    "energy_minimization_steps": 5000,
    "energy_tolerance_kj": 10.0,  # kJ/mol/nm convergence tolerance
    
    # NVT
    "equilibration_nvt_ps": 50.0, # ps for NVT equilibration
    
    # NPT
    "equilibration_npt_ps": 200.0,# ps for NPT equilibration
    "barostat_frequency": 25,     # steps between volume adjustments

    # ------------------------------------------------------------------
    # Production Settings
    # ------------------------------------------------------------------
    "production_ns": 5.0,         # nanoseconds of production MD (Research default)
    "output_frame_ps": 2.0,       # ps between saved trajectory frames

    # ------------------------------------------------------------------
    # Reproducibility & System
    # ------------------------------------------------------------------
    "seed": 42,
    "platform_preference": ["CUDA", "OpenCL", "CPU"],
    
    # ------------------------------------------------------------------
    # Output Directory
    # ------------------------------------------------------------------
    "output_base_dir": "data/md_runs",
}

def get_config_for_mode(mode: str) -> dict:
    """Returns configuration overrides for specific modes."""
    base = DEFAULT_MD_CONFIG.copy()
    base["mode"] = mode
    
    if mode == "demo":
        # Hackathon Demo Mode: Fast, approximate, just to show it works
        base.update({
            "production_ns": 0.1,        # 100 ps production
            "equilibration_nvt_ps": 20.0, # 20 ps NVT ( increased for stability)
            "equilibration_npt_ps": 50.0, # 50 ps NPT ( increased for density conv)
            "energy_minimization_steps": 1000,
            "output_frame_ps": 1.0,      # More frequent frames for smooth short viz
        })
    elif mode == "research":
        # Full Research Mode: Rigorous, publishable sampling
        base.update({
            "production_ns": 5.0,        # 5 ns production
            "equilibration_nvt_ps": 50.0,
            "equilibration_npt_ps": 200.0,
            "energy_minimization_steps": 5000,
            "output_frame_ps": 2.0,      # Standard saving interval
        })
        
    return base
