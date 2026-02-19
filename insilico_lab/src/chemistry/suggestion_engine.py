"""
Optimization suggestion engine for molecular properties.
"""
import logging

logger = logging.getLogger(__name__)

def suggest_modifications(results):
    """
    Generate optimization suggestions based on ADMET predictions.
    
    Args:
        results: Full prediction results dict
        
    Returns:
        list[str]: Suggestions for improvement
    """
    suggestions = []
    
    try:
        properties = results.get('properties', {})
        
        # Helper to extract value
        def get_val(data):
            if isinstance(data, dict):
                return data.get('value') or data.get('prediction')
            return data
        
        # Solubility suggestions
        logs = get_val(properties.get('logS'))
        if logs is not None and logs < -5:
            suggestions.append("⚠️ Low solubility detected. Consider adding polar groups (hydroxyl, amino, carboxyl)")
        
        # Lipophilicity suggestions
        logp = get_val(properties.get('logP'))
        if logp is not None:
            if logp > 5:
                suggestions.append("⚠️ High lipophilicity (logP > 5). Consider adding polar groups to reduce")
            elif logp < 0:
                suggestions.append("💡 Low lipophilicity. May have poor membrane permeability")
        
        # TPSA suggestions
        tpsa = get_val(properties.get('TPSA'))
        if tpsa is not None:
            if tpsa > 140:
                suggestions.append("⚠️ High TPSA (>140). May have poor oral bioavailability")
            elif tpsa < 20:
                suggestions.append("💡 Very low TPSA. May have CNS effects")
        
        # MW suggestions
        mw = get_val(properties.get('MW'))
        if mw is not None and mw > 500:
            suggestions.append("⚠️ High molecular weight (>500). Consider Lipinski's Rule of 5")
        
        # BBB permeability
        perm = properties.get('permeability', {})
        if isinstance(perm, dict):
            bbb = perm.get('BBB', {})
            bbb_prob = bbb.get('probability')
            
            if bbb_prob is not None and bbb_prob > 0.7:
                suggestions.append("💡 High BBB penetration. Good for CNS drugs, but check for off-target effects")
        
        # Toxicity
        toxicity = results.get('toxicity', {})
        clintox = toxicity.get('clintox', {})
        tox_prob = clintox.get('probability')
        
        if tox_prob is not None and tox_prob > 0.5:
            suggestions.append("⚠️ High toxicity risk. Consider structural modifications to reduce")
        
        # Drug-likeness
        druglikeness = results.get('druglikeness', {})
        score = druglikeness.get('score')
        
        if score is not None and score < 50:
            suggestions.append("⚠️ Low drug-likeness score. Review Lipinski and Veber rules")
        
        # Risk score
        risk = results.get('risk', {})
        risk_score = risk.get('overall_score')
        
        if risk_score is not None and risk_score > 70:
            suggestions.append("🚨 High development risk. Consider significant structural changes")
        
        # If no issues found
        if not suggestions:
            suggestions.append("✅ Molecule shows good overall ADMET profile")
        
        return suggestions
    
    except Exception as e:
        logger.error(f"Suggestion generation failed: {e}")
        return ["Unable to generate suggestions"]

if __name__ == "__main__":
    # Test with mock data
    test_results = {
        'properties': {
            'logS': -6.5,
            'logP': 5.5,
            'TPSA': 150,
            'MW': 550
        },
        'toxicity': {
            'clintox': {'probability': 0.6}
        },
        'druglikeness': {
            'score': 40
        }
    }
    
    suggestions = suggest_modifications(test_results)
    print("Suggestions:")
    for s in suggestions:
        print(f"  {s}")
