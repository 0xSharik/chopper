"""
Ranking Engine for Multi-Molecule Comparison
Provides automated scoring and ranking of drug candidates.
"""
import logging

logger = logging.getLogger(__name__)

def rank_molecules(results_list):
    """
    Rank molecules based on composite ADMET score.
    
    Args:
        results_list: List of dicts, each containing:
            - smiles: SMILES string
            - name: Molecule name (optional)
            - properties: Property predictions
            - toxicity: Toxicity predictions
            - druglikeness: Drug-likeness scores
            - risk: Risk assessment
    
    Returns:
        List of ranked molecules with scores and explanations
    """
    scored_molecules = []
    
    for idx, result in enumerate(results_list):
        try:
            score_breakdown = calculate_composite_score(result)
            
            # Identify strengths and weaknesses
            strengths, weaknesses = analyze_profile(result, score_breakdown)
            
            scored_molecules.append({
                "smiles": result.get('smiles', ''),
                "name": result.get('name', f'Molecule {idx + 1}'),
                "final_score": score_breakdown['total'],
                "score_breakdown": score_breakdown,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "raw_results": result
            })
        
        except Exception as e:
            logger.error(f"Error scoring molecule {idx}: {e}")
            scored_molecules.append({
                "smiles": result.get('smiles', ''),
                "name": result.get('name', f'Molecule {idx + 1}'),
                "final_score": 0,
                "error": str(e)
            })
    
    # Sort by score (descending)
    scored_molecules.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    
    # Assign ranks
    for rank, mol in enumerate(scored_molecules, 1):
        mol['rank'] = rank
    
    return scored_molecules

def calculate_composite_score(result):
    """Calculate weighted composite score from ADMET predictions."""
    
    # Helper to extract value
    def get_val(data):
        if isinstance(data, dict):
            return data.get('value') or data.get('prediction') or data.get('score')
        return data
    
    properties = result.get('properties', {})
    druglikeness = result.get('druglikeness', {})
    risk = result.get('risk', {})
    
    # Initialize scores
    scores = {
        'druglikeness': 0,
        'risk': 0,
        'solubility': 0,
        'lipophilicity': 0,
        'permeability': 0,
        'safety': 0,
        'penalties': 0
    }
    
    # 1. Drug-likeness (30%)
    dl_score = get_val(druglikeness.get('score'))
    if dl_score is not None:
        scores['druglikeness'] = min(100, max(0, dl_score)) * 0.30
    
    # 2. Risk (25%, inverted)
    risk_score = get_val(risk.get('overall_score'))
    if risk_score is not None:
        scores['risk'] = (100 - min(100, max(0, risk_score))) * 0.25
    
    # 3. Solubility (15%)
    logs = get_val(properties.get('logS'))
    if logs is not None:
        # Optimal: -4 to -2, normalize to 0-100
        if logs >= -2:
            sol_score = 100
        elif logs <= -8:
            sol_score = 0
        else:
            sol_score = ((logs + 8) / 6) * 100
        scores['solubility'] = sol_score * 0.15
        
        # Penalty for very low solubility
        if logs < -6:
            scores['penalties'] -= 10
    
    # 4. Lipophilicity (15%)
    logp = get_val(properties.get('logP'))
    if logp is not None:
        # Optimal: 1 to 3, normalize to 0-100
        if 1 <= logp <= 3:
            lipo_score = 100
        elif logp < 0:
            lipo_score = 0
        elif logp > 5:
            lipo_score = max(0, 100 - (logp - 5) * 20)
        else:
            lipo_score = 100 - abs(logp - 2) * 15
        scores['lipophilicity'] = lipo_score * 0.15
        
        # Penalty for high logP
        if logp > 5:
            scores['penalties'] -= 10
    
    # 5. Permeability (10%)
    perm = properties.get('permeability', {})
    if isinstance(perm, dict):
        bbb = perm.get('BBB', {})
        bbb_prob = bbb.get('probability')
        
        if bbb_prob is not None:
            scores['permeability'] = bbb_prob * 100 * 0.10
    
    # 6. Safety (5%)
    toxicity = result.get('toxicity', {})
    clintox = toxicity.get('clintox', {})
    tox_prob = clintox.get('probability')
    
    if tox_prob is not None:
        # Lower toxicity is better
        scores['safety'] = (1 - tox_prob) * 100 * 0.05
        
        # Penalty for high toxicity
        if tox_prob > 0.7:
            scores['penalties'] -= 20
    
    # Calculate total
    total = sum(scores.values())
    total = min(100, max(0, total))  # Clamp to 0-100
    
    scores['total'] = round(total, 2)
    
    return scores

def analyze_profile(result, score_breakdown):
    """Identify strengths and weaknesses of a molecule."""
    
    strengths = []
    weaknesses = []
    
    # Helper to extract value
    def get_val(data):
        if isinstance(data, dict):
            return data.get('value') or data.get('prediction') or data.get('score')
        return data
    
    properties = result.get('properties', {})
    druglikeness = result.get('druglikeness', {})
    risk = result.get('risk', {})
    
    # Drug-likeness
    dl_score = get_val(druglikeness.get('score'))
    if dl_score and dl_score > 70:
        strengths.append("High drug-likeness")
    elif dl_score and dl_score < 40:
        weaknesses.append("Low drug-likeness")
    
    # Risk
    risk_score = get_val(risk.get('overall_score'))
    if risk_score and risk_score < 30:
        strengths.append("Low development risk")
    elif risk_score and risk_score > 70:
        weaknesses.append("High development risk")
    
    # Solubility
    logs = get_val(properties.get('logS'))
    if logs and logs > -4:
        strengths.append("Good solubility")
    elif logs and logs < -6:
        weaknesses.append("Poor solubility")
    
    # Lipophilicity
    logp = get_val(properties.get('logP'))
    if logp and 1 <= logp <= 3:
        strengths.append("Optimal lipophilicity")
    elif logp and logp > 5:
        weaknesses.append("High lipophilicity")
    
    # Permeability
    perm = properties.get('permeability', {})
    if isinstance(perm, dict):
        bbb = perm.get('BBB', {})
        bbb_prob = bbb.get('probability')
        
        if bbb_prob and bbb_prob > 0.7:
            strengths.append("Good BBB penetration")
        elif bbb_prob and bbb_prob < 0.3:
            weaknesses.append("Poor BBB penetration")
    
    # Default messages
    if not strengths:
        strengths.append("Moderate profile")
    if not weaknesses:
        weaknesses.append("No major concerns")
    
    return strengths[:3], weaknesses[:3]  # Limit to top 3

def generate_executive_summary(ranked_molecules):
    """Generate executive summary for comparison results."""
    
    if not ranked_molecules:
        return "No molecules to compare."
    
    winner = ranked_molecules[0]
    
    summary_parts = []
    
    # Winner announcement
    summary_parts.append(f"🏆 **Winner: {winner['name']}** (Score: {winner['final_score']:.1f}/100)")
    summary_parts.append("")
    
    # Strengths
    if 'strengths' in winner and winner['strengths']:
        summary_parts.append(f"**Strengths:** {', '.join(winner['strengths'])}")
    
    # Recommendation
    if winner['final_score'] >= 80:
        summary_parts.append("**Recommendation:** Excellent candidate for lead optimization.")
    elif winner['final_score'] >= 60:
        summary_parts.append("**Recommendation:** Promising candidate, consider structural modifications.")
    else:
        summary_parts.append("**Recommendation:** Requires significant optimization before advancement.")
    
    return "\n".join(summary_parts)

if __name__ == "__main__":
    # Test with mock data
    test_results = [
        {
            'smiles': 'c1ccccc1',
            'name': 'Benzene',
            'properties': {'logS': -3.5, 'logP': 2.1, 'permeability': {'BBB': {'probability': 0.8}}},
            'druglikeness': {'score': 75},
            'risk': {'overall_score': 35},
            'toxicity': {'clintox': {'probability': 0.2}}
        },
        {
            'smiles': 'CCO',
            'name': 'Ethanol',
            'properties': {'logS': -0.5, 'logP': 0.3, 'permeability': {'BBB': {'probability': 0.5}}},
            'druglikeness': {'score': 45},
            'risk': {'overall_score': 50},
            'toxicity': {'clintox': {'probability': 0.3}}
        }
    ]
    
    ranked = rank_molecules(test_results)
    
    print("Ranking Results:")
    for mol in ranked:
        print(f"\n{mol['rank']}. {mol['name']}: {mol['final_score']:.1f}")
        print(f"   Strengths: {', '.join(mol.get('strengths', []))}")
        print(f"   Weaknesses: {', '.join(mol.get('weaknesses', []))}")
    
    print("\n" + "="*50)
    print(generate_executive_summary(ranked))
