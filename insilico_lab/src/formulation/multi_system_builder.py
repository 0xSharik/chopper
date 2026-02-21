"""
Multi-Molecule System Builder (Final Robust Version)
===================================================
1. Builds a SINGLE combined Topology and position list manually to avoid 
   sequential Modeller.add() unit-wrapping bugs.
2. Generates custom XMLs per molecule type including NonbondedForce parameters.
3. Uses EXACT AMBER 1-4 scale constants (0.8333333333333334 and 0.5) to avoid 
   compatibility conflicts with tip3p.xml.
4. Cleans up temp XMLs after system creation.
"""
import os
import re
import logging
import random
import numpy as np
from io import StringIO
from typing import Tuple, List

from rdkit import Chem
from rdkit.Chem import AllChem

import openmm
import openmm.app as app
import openmm.unit as unit

from src.md.system_builder import SystemBuilder

logger = logging.getLogger("md_engine")

# These MUST match exactly what amber14/tip3p.xml declares.
_AMBER_COULOMB14 = "0.8333333333333334"
_AMBER_LJ14      = "0.5"


class MultiMoleculeSystemBuilder:
    """Builds a solvated OpenMM system containing multiple solute molecules."""

    def __init__(self, config: dict):
        self.config = config
        self.base_builder = SystemBuilder(config)
        self.seed = config.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def build(
        self, mixture_list: List[dict], molecule_id: str
    ) -> Tuple[app.Topology, openmm.System, list]:
        """Build solvated multi-solute OpenMM system."""
        logger.info("═══ MultiMoleculeSystemBuilder.build() START ═══")
        
        # ── Step 1: Prepare unique molecule data ───────────────────────────
        unique_mols = []
        for idx, item in enumerate(mixture_list):
            smiles = item["smiles"]
            mol = self.base_builder._smiles_to_3d(smiles)
            mmff_props = self.base_builder._prepare_molecule_for_mmff94(mol)
            
            unique_mols.append({
                "mol": mol,
                "mmff_props": mmff_props,
                "count": item["count"],
                "resname": f"M{idx:02d}",
            })

        # ── Step 2: Box setup ──────────────────────────────────────────────
        total_solutes = sum(m["count"] for m in unique_mols)
        padding_nm = self.config.get("padding_nm", 1.5)
        if total_solutes > 20:
            padding_nm += 0.5
        box_size = max(4.0, 2.0 + padding_nm * 2)

        # ── Step 3: XML Generation (With NonbondedForce for addSolvent) ─────
        import tempfile
        temp_xmls = []
        forcefield_files = ["amber14/tip3p.xml"]

        for m_data in unique_mols:
            xml_str = self._generate_full_xml(
                m_data["mol"], m_data["mmff_props"], m_data["resname"]
            )
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".xml", mode="w")
            tf.write(xml_str)
            tf.close()
            temp_xmls.append(tf.name)
            forcefield_files.append(tf.name)

        ff = app.ForceField(*forcefield_files)

        # ── Step 4: Build TOTAL Topology and Position list ────────────────
        full_topology = app.Topology()
        full_positions = [] 
        placed_centers = []

        for m_idx, m_data in enumerate(unique_mols):
            resname = m_data["resname"]
            mol_copy = Chem.Mol(m_data["mol"])
            
            # CRITICAL: Set explicit names for index mapping in _add_bonded_forces
            for atom in mol_copy.GetAtoms():
                mi = Chem.AtomPDBResidueInfo()
                mi.SetResidueName(resname)
                mi.SetResidueNumber(m_idx + 1)
                mi.SetIsHeteroAtom(True)
                mi.SetName(f"{atom.GetSymbol()}{atom.GetIdx() + 1: <2}")
                atom.SetMonomerInfo(mi)

            # Prepare Topology Source
            pdb_block = Chem.MolToPDBBlock(mol_copy)
            pdb = app.PDBFile(StringIO(pdb_block))
            src_topo = pdb.topology
            src_pos = pdb.positions.value_in_unit(unit.nanometers)

            for i in range(m_data["count"]):
                # Distance-checked placement (Max 50 attempts)
                offset = None
                for _ in range(50):
                    trial = self._get_random_position(box_size)
                    if all(np.linalg.norm(np.array(trial) - np.array(c)) > 1.0 for c in placed_centers):
                        offset = trial
                        placed_centers.append(trial)
                        break
                
                if offset is None:
                    # Fallback to pure random if box is getting full
                    offset = self._get_random_position(box_size)

                ox, oy, oz = float(offset[0]), float(offset[1]), float(offset[2])
                
                chain = full_topology.addChain()
                residue = full_topology.addResidue(resname, chain)
                
                atom_map = {}
                for atom in src_topo.atoms():
                    new_atom = full_topology.addAtom(atom.name, atom.element, residue)
                    atom_map[atom] = new_atom
                    orig_p = src_pos[atom.index]
                    full_positions.append(openmm.Vec3(orig_p[0] + ox, orig_p[1] + oy, orig_p[2] + oz))
                
                for bond in src_topo.bonds():
                    full_topology.addBond(atom_map[bond[0]], atom_map[bond[1]])

        # ── Step 5: Modeller + Solvate ────────────────────────────────────
        # Pass full_positions as a Quantity wrapping the flat list.
        combined_modeller = app.Modeller(full_topology, full_positions * unit.nanometers)
        combined_modeller.addSolvent(ff, padding=padding_nm * unit.nanometers, model="tip3p")

        # ── Step 6: Create System ─────────────────────────────────────────
        system = ff.createSystem(
            combined_modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=0.9 * unit.nanometers,
            constraints=app.HBonds,
            hydrogenMass=4 * unit.amu,
        )

        # Cleanup temp files
        for f in temp_xmls:
            try: os.remove(f)
            except: pass

        # ── Step 7: Bonded forces ─────────────────────────────────────────
        bond_force = openmm.HarmonicBondForce()
        angle_force = openmm.HarmonicAngleForce()
        torsion_force = openmm.PeriodicTorsionForce()

        for res in combined_modeller.topology.residues():
            # Skip solvent
            if res.name in ['HOH', 'WAT', 'SOL', 'TP3']:
                continue
                
            for m_data in unique_mols:
                if res.name == m_data["resname"]:
                    # Build atom map for this residue
                    atom_mapping = {}
                    for atom in res.atoms():
                        rd_idx = self._atom_name_to_rdkit_idx(atom.name)
                        if rd_idx is not None:
                            atom_mapping[rd_idx] = atom.index
                    
                    # Append terms
                    self.base_builder._append_bonded_forces(
                        m_data["mol"], m_data["mmff_props"], atom_mapping,
                        bond_force, angle_force, torsion_force
                    )
                    break
        
        system.addForce(bond_force)
        system.addForce(angle_force)
        system.addForce(torsion_force)

        logger.info("═══ MultiMoleculeSystemBuilder.build() DONE ═══")
        return combined_modeller.topology, system, combined_modeller.positions

    def _generate_full_xml(self, mol, mmff_props, resname: str) -> str:
        """Generate XML with NonbondedForce for atom radius detection in addSolvent."""
        prefix = f"{resname}_"
        lines = ["<ForceField>", " <AtomTypes>"]
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            lines.append(f'  <Type name="{prefix}{i}" class="UNL" element="{atom.GetSymbol()}" mass="{atom.GetMass():.4f}"/>')
        lines.append(" </AtomTypes>")
        lines.append(" <Residues>")
        lines.append(f'  <Residue name="{resname}">')
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            lines.append(f'   <Atom name="{atom.GetSymbol()}{i + 1}" type="{prefix}{i}"/>')
        for bond in mol.GetBonds():
            lines.append(f'   <Bond from="{bond.GetBeginAtomIdx()}" to="{bond.GetEndAtomIdx()}"/>')
        lines.append("  </Residue>")
        lines.append(" </Residues>")
        
        # NonbondedForce WITH AMBER SCALES
        lines.append(f' <NonbondedForce coulomb14scale="{_AMBER_COULOMB14}" lj14scale="{_AMBER_LJ14}">')
        for i in range(mol.GetNumAtoms()):
            q = float(mmff_props.GetMMFFPartialCharge(i))
            vdw = mmff_props.GetMMFFVdWParams(i, i)
            sigma, epsilon = 0.3, 0.0
            if vdw:
                sigma = (vdw[0] * 0.1) * (2 ** (-1.0 / 6.0))
                epsilon = vdw[1] * 4.184
            lines.append(f'  <Atom type="{prefix}{i}" charge="{q:.6f}" sigma="{sigma:.6f}" epsilon="{epsilon:.6f}"/>')
        lines.append(" </NonbondedForce>")
        lines.append("</ForceField>")
        return "\n".join(lines)

    def _atom_name_to_rdkit_idx(self, atom_name: str):
        match = re.search(r"(\d+)", atom_name)
        return int(match.group(1)) - 1 if match else None

    def _add_bonded_forces_to_residue(self, system, mol, mmff_props, residue):
        atom_mapping = {}
        for atom in residue.atoms():
            rd_idx = self._atom_name_to_rdkit_idx(atom.name)
            if rd_idx is not None:
                atom_mapping[rd_idx] = atom.index
        self.base_builder._add_bonded_forces(system, mol, mmff_props, residue.chain.topology, atom_map=atom_mapping)

    def _get_random_position(self, box_size: float) -> openmm.Vec3:
        return openmm.Vec3(random.uniform(0, box_size), random.uniform(0, box_size), random.uniform(0, box_size))
