"""
Comparison utilities for molecular property analysis.
"""
import pandas as pd

def compare_properties(base_results, modified_results):
    """
    Compare properties between base and modified molecules.
    
    Returns:
        dict: Property deltas and comparisons
    """
    deltas = {}
    
    # Helper to extract numeric value
    def get_val(data):
        if isinstance(data, dict):
            return data.get('value') or data.get('prediction')
        return data
    
    # Compare basic properties
    props_to_compare = ['MW', 'logP', 'logS', 'logD', 'TPSA', 'HBD', 'HBA']
    
    for prop in props_to_compare:
        base_val = get_val(base_results.get(prop))
        mod_val = get_val(modified_results.get(prop))
        
        if base_val is not None and mod_val is not None:
            delta = mod_val - base_val
            pct_change = (delta / base_val * 100) if base_val != 0 else 0
            deltas[prop] = {
                'base': base_val,
                'modified': mod_val,
                'delta': delta,
                'pct_change': pct_change
            }
    
    # Compare permeability
    base_perm = base_results.get('permeability', {})
    mod_perm = modified_results.get('permeability', {})
    
    if isinstance(base_perm, dict) and isinstance(mod_perm, dict):
        # Caco2
        base_caco2 = get_val(base_perm.get('Caco2', {}).get('prediction'))
        mod_caco2 = get_val(mod_perm.get('Caco2', {}).get('prediction'))
        if base_caco2 and mod_caco2:
            deltas['Caco2'] = {
                'base': base_caco2,
                'modified': mod_caco2,
                'delta': mod_caco2 - base_caco2
            }
        
        # BBB
        base_bbb = base_perm.get('BBB', {}).get('probability')
        mod_bbb = mod_perm.get('BBB', {}).get('probability')
        if base_bbb is not None and mod_bbb is not None:
            deltas['BBB'] = {
                'base': base_bbb,
                'modified': mod_bbb,
                'delta': mod_bbb - base_bbb
            }
    
    return deltas

def generate_comparison_table(base_results, modified_results):
    """Generate formatted comparison table for Streamlit."""
    deltas = compare_properties(base_results, modified_results)
    
    rows = []
    for prop, data in deltas.items():
        rows.append({
            'Property': prop,
            'Original': f"{data['base']:.2f}" if isinstance(data['base'], (int, float)) else str(data['base']),
            'Modified': f"{data['modified']:.2f}" if isinstance(data['modified'], (int, float)) else str(data['modified']),
            'Change': f"{data['delta']:+.2f}" if isinstance(data['delta'], (int, float)) else str(data['delta'])
        })
    
    return pd.DataFrame(rows)

def generate_delta_summary(base_results, modified_results):
    """Generate human-readable summary of changes."""
    deltas = compare_properties(base_results, modified_results)
    
    summary_parts = []
    
    # LogP changes
    if 'logP' in deltas:
        delta = deltas['logP']['delta']
        if delta > 0.5:
            summary_parts.append("increased lipophilicity")
        elif delta < -0.5:
            summary_parts.append("decreased lipophilicity")
    
    # LogS changes
    if 'logS' in deltas:
        delta = deltas['logS']['delta']
        if delta > 0.5:
            summary_parts.append("improved solubility")
        elif delta < -0.5:
            summary_parts.append("reduced solubility")
    
    # MW changes
    if 'MW' in deltas:
        delta = deltas['MW']['delta']
        if abs(delta) > 10:
            summary_parts.append(f"{'increased' if delta > 0 else 'decreased'} molecular weight")
    
    # TPSA changes
    if 'TPSA' in deltas:
        delta = deltas['TPSA']['delta']
        if delta > 10:
            summary_parts.append("increased polarity")
        elif delta < -10:
            summary_parts.append("decreased polarity")
    
    # BBB changes
    if 'BBB' in deltas:
        delta = deltas['BBB']['delta']
        if abs(delta) > 0.1:
            summary_parts.append(f"{'improved' if delta > 0 else 'reduced'} BBB penetration")
    
    if summary_parts:
        return "Modification " + ", ".join(summary_parts) + "."
    else:
        return "Minimal property changes observed."

def get_delta_color(delta, property_name):
    """
    Determine color for delta based on property and direction.
    
    Returns:
        str: 'green', 'red', or 'gray'
    """
    # Positive changes (green)
    if property_name in ['logS', 'BBB'] and delta > 0:
        return 'green'
    
    # Negative changes that are bad (red)
    if property_name == 'logP' and delta > 1:  # Too lipophilic
        return 'red'
    if property_name == 'logS' and delta < -1:  # Less soluble
        return 'red'
    if property_name == 'MW' and delta > 100:  # Too heavy
        return 'red'
    
    # Neutral
    return 'gray'
