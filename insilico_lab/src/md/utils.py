"""
utils.py — Utility functions for the MD module.

Provides:
  - GPU/CPU platform detection (CUDA → OpenCL → CPU fallback)
  - Structured logging to md.log with timestamps
  - Safe directory creation
  - OpenMM unit shorthands for clarity throughout the module
"""

import os
import logging
from logging.handlers import RotatingFileHandler

# OpenMM imports (guarded so module can be imported even before openmm install
# during environment checks — hard-import in callers only when running sim)
try:
    import openmm
    import openmm.unit as unit
    from openmm import Platform
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Unit shorthands (for callers that import from utils)
# ---------------------------------------------------------------------------
if OPENMM_AVAILABLE:
    kelvin        = unit.kelvin
    picoseconds   = unit.picoseconds
    femtoseconds  = unit.femtoseconds
    nanometers    = unit.nanometers
    atmospheres   = unit.atmospheres
    angstroms     = unit.angstroms
    kilojoules_per_mole = unit.kilojoules_per_mole
    per_picosecond      = unit.picoseconds**-1
    molar               = unit.molar


# ---------------------------------------------------------------------------
# Platform / GPU detection
# ---------------------------------------------------------------------------
def get_best_platform():
    """
    Detect and return the fastest available OpenMM Platform.

    Priority: CUDA > OpenCL > CPU
    Returns a tuple (Platform object, platform_name: str).
    """
    if not OPENMM_AVAILABLE:
        raise RuntimeError("OpenMM is not installed.")

    platform_priority = ["CUDA", "OpenCL", "CPU"]
    for name in platform_priority:
        try:
            platform = Platform.getPlatformByName(name)
            # Quick sanity check — will raise if platform unavailable on hw
            _ = platform.getName()
            return platform, name
        except Exception:
            continue

    # Should never reach here — CPU is always available
    raise RuntimeError("No OpenMM platform found (not even CPU). Check OpenMM install.")


def get_platform_properties(platform_name: str) -> dict:
    """
    Return platform-specific property dict for Simulation constructor.
    CUDA uses mixed precision for speed; others use defaults.
    """
    if platform_name == "CUDA":
        return {"CudaPrecision": "mixed"}
    elif platform_name == "OpenCL":
        return {"OpenCLPrecision": "mixed"}
    return {}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(run_dir: str = None, logger_name: str = "md_engine") -> logging.Logger:
    """
    Create (or retrieve) a logger.
    If run_dir is provided, adds a file handler for md.log.
    Always checks/adds a console handler.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent duplication by root logger

    # Avoid duplicate handlers if called multiple times?
    # Actually we might want to ADD a file handler if we only had console before.
    # But for simplicity, if handlers exist, just return? 
    # No, if we transition from Console->File, we want to add File.
    
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in logger.handlers)
    has_file = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
    
    fmt_str = "[%(asctime)s] %(levelname)-8s  %(message)s"
    fmt = logging.Formatter(fmt_str, datefmt="%Y-%m-%d %H:%M:%S")

    # Add Console if missing
    if not has_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # Add File if run_dir provided and missing
    if run_dir:
        ensure_dir(run_dir)
        log_path = os.path.join(run_dir, "md.log")
        # Check if this specific file handler exists? 
        # Hard to check path. Just check if ANY file handler exists?
        # A bit simplistic but okay for now.
        if not has_file:
            fh = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=3)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it does not exist. Returns path."""
    os.makedirs(path, exist_ok=True)
    return path


def get_run_dir(base_dir: str, molecule_id: str) -> str:
    """
    Construct and create the output directory for a given molecule run.
    Returns the absolute path.
    """
    run_dir = os.path.join(base_dir, molecule_id)
    ensure_dir(run_dir)
    return os.path.abspath(run_dir)
