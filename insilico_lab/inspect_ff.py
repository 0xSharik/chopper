
from openmm.app import ForceField
try:
    ff = ForceField('amber14/tip3p.xml')
    if hasattr(ff, '_forces'):
        gens = ff._forces
        for gen in gens:
             name = type(gen).__name__
             if "NonbondedGenerator" in name:
                 print(f"Generator: {name}")
                 if hasattr(gen, 'coulomb14scale'):
                     print(f"Coulomb14: {gen.coulomb14scale}")
                     print(f"Type: {type(gen.coulomb14scale)}")
                 if hasattr(gen, 'lj14scale'):
                     print(f"LJ14: {gen.lj14scale}")
except Exception as e:
    print(f"Error: {e}")
