"""
src/md/__init__.py

Public API for the MD module.
GAFF2 force field (openmmforcefields) + TIP3P explicit solvent.
"""

from src.md.md_engine import MolecularDynamicsEngine
from src.md.config   import DEFAULT_MD_CONFIG

__all__ = ["MolecularDynamicsEngine", "DEFAULT_MD_CONFIG"]
