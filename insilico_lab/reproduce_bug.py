import os, sys
import openmm
import openmm.app as app
import openmm.unit as unit
from io import StringIO
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. Setup minimal forcefield
# Just tip3p and a tiny custom topology
ff_xml = """<ForceField>
 <AtomTypes>
  <Type name="M00_0" class="UNL" element="C" mass="12.011"/>
 </AtomTypes>
 <Residues>
  <Residue name="M00">
   <Atom name="C1" type="M00_0"/>
  </Residue>
 </Residues>
</ForceField>"""

import tempfile
tf = tempfile.NamedTemporaryFile(delete=False, suffix=".xml", mode="w")
tf.write(ff_xml)
tf.close()

ff = app.ForceField("amber14/tip3p.xml", tf.name)

# 2. Build topology/positions manually (same as our builder)
top = app.Topology()
chain = top.addChain()
res = top.addResidue("M00", chain)
elem = app.Element.getBySymbol("C")
top.addAtom("C1", elem, res)

pos = [openmm.Vec3(0,0,0)] * unit.nanometers

# 3. Create Modeller
mod = app.Modeller(top, pos)
print("Modeller created OK")

# 4. Solvate (this calls Modeller.add internally)
print("Calling addSolvent...")
try:
    mod.addSolvent(ff, model="tip3p", padding=1.0*unit.nanometers)
    print("addSolvent OK")
except Exception as e:
    print(f"FAILED addSolvent: {e}")
    import traceback
    traceback.print_exc()

# 5. Create System
print("Calling createSystem...")
try:
    system = ff.createSystem(
        mod.topology,
        nonbondedMethod=app.PME,
        constraints=app.HBonds
    )
    print("createSystem OK")
except Exception as e:
    print(f"FAILED createSystem: {e}")
    import traceback
    traceback.print_exc()

os.remove(tf.name)
