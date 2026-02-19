"""
verify_install.py — Verify the GAFF2-based MD stack.

No openff imports. Pure: RDKit + OpenMM + openmmforcefields + mdtraj.
"""
import sys
print(f"Python: {sys.version.split()[0]}")
print()

results = []
checks = [
    ("openmm",              "import openmm; print(openmm.__version__)"),
    ("openmmforcefields",   "from openmmforcefields.generators import GAFFTemplateGenerator; print('OK')"),
    ("mdtraj",              "import mdtraj; print(mdtraj.__version__)"),
    ("rdkit",               "from rdkit import Chem; from rdkit.Chem import AllChem; print('OK')"),
    ("pandas",              "import pandas; print(pandas.__version__)"),
    ("numpy",               "import numpy; print(numpy.__version__)"),
    ("scikit-learn",        "import sklearn; print(sklearn.__version__)"),
    ("xgboost",             "import xgboost; print(xgboost.__version__)"),
    ("deepchem",            "import deepchem; print(deepchem.__version__)"),
    ("streamlit",           "import streamlit; print(streamlit.__version__)"),
]

for name, code in checks:
    try:
        exec(code)
        print(f"  [PASS] {name}")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        results.append(False)

print()
print(f"{sum(results)}/{len(results)} imports OK")
if all(results):
    print("Stack ready. Run: python test_md_pipeline.py")
