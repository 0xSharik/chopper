"""
Narrative Engine for Executive Summaries
Generates professional, confident summaries of ADMET predictions.
"""

def generate_executive_summary(results):
    """
    Generate executive summary from ADMET predictions.
    
    Args:
        results: Dict containing properties, toxicity, druglikeness, risk
        
    Returns:
        str: Professional 3-4 sentence summary
    """
    # Helper to extract values
    def get_val(data):
        if isinstance(data, dict):
            return data.get('value') or data.get('prediction') or data.get('score')
        return data
    
    properties = results.get('properties', {})
    druglikeness = results.get('druglikeness', {})
    risk = results.get('risk', {})
    toxicity = results.get('toxicity', {})
    
    summary_parts = []
    
    # 1. Lipophilicity assessment
    logp = get_val(properties.get('logP'))
    if logp is not None:
        if logp > 5:
            summary_parts.append(f"High lipophilicity (logP: {logp:.1f}) may limit oral bioavailability and increase off-target binding.")
        elif 1 <= logp <= 3:
            summary_parts.append(f"Optimal lipophilicity (logP: {logp:.1f}) suggests good membrane permeability and oral absorption.")
        elif logp < 1:
            summary_parts.append(f"Low lipophilicity (logP: {logp:.1f}) may result in poor membrane penetration.")
    
    # 2. Solubility assessment
    logs = get_val(properties.get('logS'))
    if logs is not None:
        if logs < -5:
            summary_parts.append(f"Poor aqueous solubility (logS: {logs:.1f}) requires formulation optimization or prodrug strategy.")
        elif logs > -4:
            summary_parts.append(f"Good aqueous solubility (logS: {logs:.1f}) facilitates formulation development.")
    
    # 3. BBB penetration
    perm = properties.get('permeability', {})
    if isinstance(perm, dict):
        bbb = perm.get('BBB', {})
        bbb_prob = bbb.get('probability')
        
        if bbb_prob is not None:
            if bbb_prob > 0.7:
                summary_parts.append(f"Strong CNS penetration potential (BBB: {bbb_prob:.1%}) suitable for neurological indications.")
            elif bbb_prob < 0.3:
                summary_parts.append(f"Limited CNS penetration (BBB: {bbb_prob:.1%}) reduces off-target CNS effects.")
    
    # 4. Toxicity concerns
    clintox = toxicity.get('clintox', {})
    tox_prob = clintox.get('probability')
    
    if tox_prob is not None:
        if tox_prob > 0.7:
            summary_parts.append(f"Elevated toxicity risk ({tox_prob:.1%}) warrants structural modification to improve safety profile.")
        elif tox_prob < 0.3:
            summary_parts.append(f"Low predicted toxicity ({tox_prob:.1%}) supports favorable safety profile.")
    
    # 5. Overall assessment
    dl_score = get_val(druglikeness.get('score'))
    risk_score = get_val(risk.get('overall_score'))
    
    if dl_score is not None and risk_score is not None:
        if dl_score > 75 and risk_score < 40:
            summary_parts.append("**Recommendation:** Excellent lead candidate for optimization and advancement.")
        elif dl_score > 60 and risk_score < 60:
            summary_parts.append("**Recommendation:** Promising candidate; consider targeted structural modifications.")
        elif dl_score < 40 or risk_score > 70:
            summary_parts.append("**Recommendation:** Requires significant optimization before clinical development.")
        else:
            summary_parts.append("**Recommendation:** Moderate profile; evaluate alternative scaffolds in parallel.")
    
    # Default if no data
    if not summary_parts:
        summary_parts.append("Insufficient data for comprehensive assessment. Additional experimental validation recommended.")
    
    # Limit to 4 sentences for conciseness
    return " ".join(summary_parts[:4])

def generate_property_insight(property_name, value, optimal_range):
    """
    Generate specific insight for a single property.
    
    Args:
        property_name: Name of property
        value: Current value
        optimal_range: Tuple of (min, max) optimal values
        
    Returns:
        str: Insight message
    """
    min_opt, max_opt = optimal_range
    
    insights = {
        'MW': {
            'optimal': "Molecular weight within Lipinski's Rule of 5",
            'high': "High molecular weight may reduce oral bioavailability",
            'low': "Low molecular weight may lack specificity"
        },
        'logP': {
            'optimal': "Lipophilicity balanced for membrane permeability",
            'high': "Excessive lipophilicity may cause poor solubility",
            'low': "Low lipophilicity may limit membrane penetration"
        },
        'TPSA': {
            'optimal': "Polar surface area suitable for oral absorption",
            'high': "High polarity may reduce membrane permeability",
            'low': "Low polarity may increase off-target binding"
        }
    }
    
    if min_opt <= value <= max_opt:
        return insights.get(property_name, {}).get('optimal', 'Within optimal range')
    elif value > max_opt:
        return insights.get(property_name, {}).get('high', 'Above optimal range')
    else:
        return insights.get(property_name, {}).get('low', 'Below optimal range')

if __name__ == "__main__":
    # Test with mock data
    test_results = {
        'properties': {
            'logS': -3.5,
            'logP': 2.1,
            'TPSA': 85,
            'MW': 284,
            'permeability': {
                'BBB': {'probability': 0.8}
            }
        },
        'druglikeness': {'score': 78},
        'risk': {'overall_score': 35},
        'toxicity': {'clintox': {'probability': 0.2}}
    }
    
    summary = generate_executive_summary(test_results)
    print("Executive Summary:")
    print(summary)
