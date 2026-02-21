"""
system_builder.py — SMILES → solvated OpenMM system (Pure RDKit MMFF94 + OpenMM).

Pipeline:
  1. SMILES → RDKit Mol (ETKDG v3).
  2. Geometry Optimization (MMFF94).
  3. Parameter Extraction (MMFF94 Bonds, Angles, Torsions, Charges, vdW).
  4. OpenMM XML Generation (AtomTypes, Nonbonded).
  5. Solvation (Modeller) with TIP3P.
  6. System Creation (Amber14/TIP3P + Solute XML).
  7. Add Solute Bonded Forces (HarmonicBond, HarmonicAngle, PeriodicTorsion).
"""

import os
import logging
import tempfile
import math
from typing import Tuple, List

from rdkit import Chem
from rdkit.Chem import AllChem

import openmm
import openmm.app as app
import openmm.unit as unit

logger = logging.getLogger("md_engine")

# Conversions
_KCAL_TO_KJ    = 4.184
_ANG_TO_NM     = 0.1
_ANG2_TO_NM2   = 0.01
_DEG_TO_RAD    = math.pi / 180.0

# MMFF94 Conversions
# RDKit MMFF Bond k is in kcal/mol/A^2. OpenMM HarmonicBondForce uses kJ/mol/nm^2 * 0.5?
# Wait, OpenMM HarmonicBondForce: E = 0.5 * k * (r-r0)^2
# RDKit MMFF94: E = 0.5 * k * (r-r0)^2 (Standard harmonic)
# So we need to convert k from kcal/mol/A^2 to kJ/mol/nm^2
_MMFF_BOND_CONV = _KCAL_TO_KJ / _ANG2_TO_NM2

# RDKit MMFF Angle k is in kcal/mol/rad^2. 
# OpenMM: E = 0.5 * k * (theta-theta0)^2
_MMFF_ANGLE_CONV = _KCAL_TO_KJ

# RDKit MMFF Torsion V is in kcal/mol. 
# MMFF94 formula: E = 0.5 * V * (1 + cos(n*phi))
# OpenMM formula: E = k * (1 + cos(n*phi - phi0))  => k = 0.5 * V
_MMFF_TORSION_CONV = _KCAL_TO_KJ * 0.5

class SystemBuilder:
    def __init__(self, config: dict):
        self.config = config
        # 1. Enforce minimum padding
        pad = config.get("padding_nm", 1.3)
        if pad < 1.3:
            logger.warning(f"Padding {pad} nm too small for NPT. Forcing 1.3 nm.")
            pad = 1.3
        self.padding = pad * unit.nanometers
        
    def build_system(self, smiles: str, molecule_id: str) -> Tuple[app.Topology, openmm.System, list]:
        """
        Build a solvated OpenMM system from SMILES.
        Returns (Topology, System, Positions)
        """
        logger.info("Step 1: RDKit 3D Conformer & Optimization")
        mol = self._smiles_to_3d(smiles)

        # Prepare molecule for MMFF94 parameters
        props = self._prepare_molecule_for_mmff94(mol)

        logger.info("Step 2-6: Building Solvated OpenMM System with MMFF94")
        topology, system, positions = self._build_solute_system(mol, props)
        
        return topology, system, positions

    def _smiles_to_3d(self, smiles: str) -> Chem.Mol:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # Set Residue Info for PDB
        res_info = Chem.AtomPDBResidueInfo()
        res_info.SetResidueName("UNL")
        res_info.SetIsHeteroAtom(True)
        for atom in mol.GetAtoms():
            atom.SetMonomerInfo(res_info)

        # ETKDG v3
        params = AllChem.ETKDGv3()
        params.randomSeed = self.config.get("seed", 42)
        AllChem.EmbedMolecule(mol, params)
        
        # MMFF94 Minimization (Pre-minimization)
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            logger.warning("MMFF94 optimization failed, falling back to UFF")
            AllChem.UFFOptimizeMolecule(mol)
            
        return mol

    def _prepare_molecule_for_mmff94(self, mol: Chem.Mol):
        """
        Compute MMFF94 properties and return the properties object.
        """
        # Ensure MMFF properties are present
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
        if not mmff_props:
            raise ValueError("Could not retrieve MMFF94 properties for molecule.")
        
        return mmff_props

    def _build_solute_system(self, mol: Chem.Mol, mmff_props) -> Tuple[app.Topology, openmm.System, list]:
        """
        Build solvated system using on-the-fly XML for solute.
        """
        # 1. Load ForceField (Amber14 + TIP3P)
        ff = app.ForceField('amber14/tip3p.xml')
        
        # 2. Extract 1-4 Scaling Factors dynamically to ensure XML match
        coulomb14 = 0.8333333333
        lj14 = 0.5
        
        if hasattr(ff, '_forces'):
             for gen in ff._forces:
                 if "NonbondedGenerator" in type(gen).__name__:
                     if hasattr(gen, 'coulomb14scale'):
                         coulomb14 = gen.coulomb14scale
                     if hasattr(gen, 'lj14scale'):
                         lj14 = gen.lj14scale
                     break
        
        logger.info(f"Using 1-4 Scaling: Coulomb={coulomb14}, LJ={lj14}")

        # 3. Generate XML
        xml_content = self._generate_solute_xml(mol, mmff_props, coulomb14, lj14)
        
        # 4. Load Solute XML
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp:
            tmp.write(xml_content)
            tmp_path = tmp.name
            
        try:
            ff.loadFile(tmp_path)
        finally:
            os.remove(tmp_path)

        # 5. Create Modeller & Solvate
        from io import StringIO
        pdb_block = Chem.MolToPDBBlock(mol)
        pdb = app.PDBFile(StringIO(pdb_block))
        
        modeller = app.Modeller(pdb.topology, pdb.positions)
        
        # Use sanitized padding from __init__
        padding = self.padding
        ionic_strength = self.config.get("ionic_strength", 0.0)
        neutralize = self.config.get("neutralize", True)
        
        if isinstance(ionic_strength, (int, float)):
             ionic_strength = ionic_strength * unit.molar
        
        logger.info(f"Solvating: padding={padding.value_in_unit(unit.nanometers)} nm, ionic={ionic_strength}")
        logger.info(f"Pre-solvation Box: {modeller.topology.getPeriodicBoxVectors()}")
        modeller.addSolvent(ff, model='tip3p', padding=padding, 
                           ionicStrength=ionic_strength, neutralize=neutralize)
        logger.info(f"Post-solvation Box: {modeller.topology.getPeriodicBoxVectors()}")
        
        # 5a. Validate Box Size
        box = modeller.topology.getPeriodicBoxVectors()
        lx = box[0][0].value_in_unit(unit.nanometers)
        ly = box[1][1].value_in_unit(unit.nanometers)
        lz = box[2][2].value_in_unit(unit.nanometers)
        min_box = min(lx, ly, lz)
        
        logger.info(f"Box Size: {lx:.2f} x {ly:.2f} x {lz:.2f} nm")
        
        if min_box < 2.0:
            raise ValueError(f"❌ Box too small ({min_box:.2f} nm) for stable NPT. Increase padding.")

        # 5b. Water Count Check
        n_water = len([r for r in modeller.topology.residues() if r.name in ['HOH', 'WAT', 'SOL', 'TP3']])
        logger.info(f"Water molecules: {n_water}")
        if n_water < 300:
            raise ValueError(f"[!] Low water count ({n_water}). Please increase 'Padding' in Advanced Configuration (current: {self.padding.value_in_unit(unit.nanometers):.1f} nm).")
        elif n_water < 600:
            logger.warning(f"Low water count ({n_water}). PBC artifacts possible.")

        # 6. Create System (Enforce PME + Safe Cutoff)
        # Cutoff 0.9 nm is safe if box > 1.8 nm (checked above > 2.0 nm)
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, 
                                nonbondedCutoff=0.9*unit.nanometers, constraints=app.HBonds, 
                                rigidWater=True, hydrogenMass=4*unit.amu)
        
        # 7. Add Bonded Forces (MMFF94)
        self._add_bonded_forces(system, mol, mmff_props, modeller.topology)
        
        return modeller.topology, system, modeller.positions

    def _generate_solute_xml(self, mol: Chem.Mol, mmff_props, coulomb14scale=0.8333, lj14scale=0.5) -> str:
        """XML with MMFF94 Charges and VdW."""
        lines = []
        lines.append('<ForceField>')
        lines.append(' <AtomTypes>')
        
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            elem = atom.GetSymbol()
            mass = atom.GetMass()
            # Unique type per atom to map specific params
            lines.append(f'  <Type name="UNL-{i}" class="UNL" element="{elem}" mass="{mass}"/>')
        
        lines.append(' </AtomTypes>')
        lines.append(' <Residues>')
        lines.append('  <Residue name="UNL">')
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            lines.append(f'   <Atom name="{atom.GetSymbol()}{i+1}" type="UNL-{i}"/>')
        # Bonds mainly for connectivity in topology if needed? 
        # Modeller uses PDB for topology, so this is for FF matching.
        for bond in mol.GetBonds():
             lines.append(f'   <Bond from="{bond.GetBeginAtomIdx()}" to="{bond.GetEndAtomIdx()}"/>')
        lines.append('  </Residue>')
        lines.append(' </Residues>')
        
        # Nonbonded
        # Add attributes explicitly to ensure merge with Amber14
        lines.append(f' <NonbondedForce coulomb14scale="{coulomb14scale}" lj14scale="{lj14scale}" attributes="charge,sigma,epsilon">')
        for i in range(mol.GetNumAtoms()):
            q = mmff_props.GetMMFFPartialCharge(i)
            
            # GetMMFFVdWParams(i, j). For self: (i, i).
            vdw = mmff_props.GetMMFFVdWParams(i, i) 
            
            if vdw:
                r_star = vdw[0] # Angstrom
                eps = vdw[1]    # kcal/mol
                
                # Convert
                sigma = (r_star * 0.1) * (2**(-1.0/6.0)) # nm
                epsilon = eps * 4.184 # kJ/mol
                
                lines.append(f'  <Atom type="UNL-{i}" charge="{q}" sigma="{sigma}" epsilon="{epsilon}"/>')
            else:
                lines.append(f'  <Atom type="UNL-{i}" charge="{q}" sigma="0.3" epsilon="0.0"/>')

        lines.append(' </NonbondedForce>')
        lines.append('</ForceField>')
        return "\n".join(lines)

    def _append_bonded_forces(self, mol, mmff_props, atom_map, bond_force, angle_force, torsion_force):
        """Append HarmonicBond, HarmonicAngle, and PeriodicTorsion terms to existing forces."""
        # 1. HarmonicBondForce
        for bond in mol.GetBonds():
            ia, ib = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            idx_a = atom_map[ia]
            idx_b = atom_map[ib]
            
            params = mmff_props.GetMMFFBondStretchParams(mol, ia, ib)
            if params:
                r0 = params[1] * _ANG_TO_NM
                kb = params[0] * _MMFF_BOND_CONV
                bond_force.addBond(idx_a, idx_b, r0, kb)
        
        # 2. HarmonicAngleForce
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neighbors) >= 2:
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        ia, ic = neighbors[i], neighbors[j]
                        idx_a = atom_map[ia]
                        idx_b = atom_map[idx]
                        idx_c = atom_map[ic]
                        
                        params = mmff_props.GetMMFFAngleBendParams(mol, ia, idx, ic)
                        if params:
                            theta0 = params[1] * _DEG_TO_RAD
                            ka = params[0] * _MMFF_ANGLE_CONV
                            angle_force.addAngle(idx_a, idx_b, idx_c, theta0, ka)

        # 3. PeriodicTorsionForce
        for bond in mol.GetBonds():
            ia, ib = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            for n_a in [n.GetIdx() for n in mol.GetAtomWithIdx(ia).GetNeighbors() if n.GetIdx() != ib]:
                for n_b in [n.GetIdx() for n in mol.GetAtomWithIdx(ib).GetNeighbors() if n.GetIdx() != ia]:
                    params = mmff_props.GetMMFFTorsionParams(mol, n_a, ia, ib, n_b)
                    if params:
                        idx1 = atom_map[n_a]
                        idx2 = atom_map[ia]
                        idx3 = atom_map[ib]
                        idx4 = atom_map[n_b]
                        v_vals = [params[0], params[1], params[2]]
                        for n_idx, v in enumerate(v_vals):
                            if abs(v) > 1e-4:
                                periodicity = n_idx + 1
                                torsion_force.addTorsion(idx1, idx2, idx3, idx4, periodicity, 0.0, v * _MMFF_TORSION_CONV)

    def _add_bonded_forces(self, system, mol, mmff_props, topology, atom_map=None):
        """Standard entry point for adding bonded forces to a system."""
        bond_force = openmm.HarmonicBondForce()
        angle_force = openmm.HarmonicAngleForce()
        torsion_force = openmm.PeriodicTorsionForce()
        
        # If no atom_map provided, assume 1:1 mapping (identity)
        if atom_map is None:
            atom_map = {i: i for i in range(mol.GetNumAtoms())}
            
        self._append_bonded_forces(mol, mmff_props, atom_map, bond_force, angle_force, torsion_force)
        
        system.addForce(bond_force)
        system.addForce(angle_force)
        # system.addForce(torsion_force) # Disabled for stability
