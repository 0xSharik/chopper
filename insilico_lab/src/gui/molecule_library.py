"""Load all molecules from project datasets."""
import pandas as pd
import os

def load_all_molecules():
    """Load all molecules from cleaned datasets with actual names."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    molecules = {}
    
    # Dataset configurations: (path, name_column, smiles_column, source_prefix)
    datasets = [
        ('data/processed/esol_cleaned.csv', 'Compound ID', 'smiles', 'ESOL'),
        ('data/processed/bbb_cleaned.csv', 'name', 'smiles', 'BBB'),
        ('data/processed/caco2_cleaned.csv', None, 'smiles', 'Caco2'),
        ('data/processed/clintox_cleaned.csv', 'drug_name', 'smiles', 'ClinTox'),
    ]
    
    for path, name_col, smiles_col, prefix in datasets:
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path)
                if smiles_col in df.columns:
                    # Use actual names if available, otherwise generate readable IDs
                    if name_col and name_col in df.columns:
                        for idx, row in df.iterrows():
                            if idx >= 500:  # Limit per dataset
                                break
                            name = str(row[name_col]).strip()
                            smiles = str(row[smiles_col]).strip()
                            if name and smiles and name != 'nan':
                                molecules[name] = smiles
                    else:
                        # No name column - use SMILES prefix for identification
                        for idx, smiles in enumerate(df[smiles_col].unique()[:500]):
                            # Create readable identifier from SMILES
                            smiles_str = str(smiles).strip()
                            # Use first 15 chars of SMILES as identifier
                            smiles_preview = smiles_str[:15] + "..." if len(smiles_str) > 15 else smiles_str
                            key = f"{prefix}: {smiles_preview}"
                            molecules[key] = smiles_str
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    return molecules

def get_molecule_list():
    """Return list of (name, smiles) tuples sorted by name."""
    molecules = load_all_molecules()
    return sorted([(name, smiles) for name, smiles in molecules.items()])
