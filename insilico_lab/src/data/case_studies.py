"""
Case Study Presets for Demo
One-click molecule loading for presentations.
"""

PRESET_CASES = {
    "Diazepam (CNS Drug)": {
        "smiles": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",
        "description": "💊 Benzodiazepine anxiolytic - demonstrates strong CNS penetration and drug-like properties",
        "category": "CNS"
    },
    "Phenol (Benchmark)": {
        "smiles": "Oc1ccccc1",
        "description": "Benchmark compound - standard for validation and testing",
        "category": "Reference"
    },
    "Ciprofloxacin (Antibiotic)": {
        "smiles": "C1CC1N2C=C(C(=O)c3cc(F)c(cc23)N4CCNCC4)C(=O)O",
        "description": "🦠 Fluoroquinolone antibiotic - complex structure with multiple pharmacophores",
        "category": "Antibiotic"
    },
    "Aspirin (Baseline)": {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "description": "⭐ Classic NSAID - simple, well-characterized reference compound",
        "category": "Pain Relief"
    },
    "Acetaminophen (Analgesic)": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "description": "💊 Common pain reliever - excellent RDKit logP agreement",
        "category": "Pain Relief"
    },
    "Atorvastatin (Statin)": {
        "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(c(c(c1)c3ccccc3)C(=O)O)c4ccc(F)cc4",
        "description": "💊 HMG-CoA reductase inhibitor - cholesterol-lowering blockbuster drug",
        "category": "Cardiovascular"
    },
    "Metformin (Antidiabetic)": {
        "smiles": "CN(C)C(=N)NC(=N)N",
        "description": "💉 Biguanide antidiabetic - first-line treatment for type 2 diabetes",
        "category": "Metabolic"
    },
    "Warfarin (Anticoagulant)": {
        "smiles": "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O",
        "description": "🩸 Vitamin K antagonist - demonstrates complex metabolism profile",
        "category": "Cardiovascular"
    },
    "Morphine (Opioid)": {
        "smiles": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
        "description": "💊 Opioid analgesic - strong CNS activity, high BBB penetration",
        "category": "Pain Relief"
    },
    "Penicillin G (Antibiotic)": {
        "smiles": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
        "description": "🦠 Beta-lactam antibiotic - historic breakthrough in medicine",
        "category": "Antibiotic"
    }
}

def get_case_study_names():
    """Return list of case study names."""
    return list(PRESET_CASES.keys())

def get_case_study(name):
    """Get case study details by name."""
    return PRESET_CASES.get(name)

def get_cases_by_category(category):
    """Get all case studies in a category."""
    return {
        name: details 
        for name, details in PRESET_CASES.items() 
        if details['category'] == category
    }

if __name__ == "__main__":
    print("Available Case Studies:")
    for name, details in PRESET_CASES.items():
        print(f"\n{name}")
        print(f"  Category: {details['category']}")
        print(f"  SMILES: {details['smiles']}")
        print(f"  {details['description']}")
