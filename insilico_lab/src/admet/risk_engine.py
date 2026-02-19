
import os
import sys
import logging
import numpy as np

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class RiskEngine:
    """Aggregates various risk factors into an overall development risk score."""
    
    def __init__(self):
        # Weights for different risk components
        self.weights = {
            'toxicity': 0.35,
            'permeability': 0.20,
            'druglikeness': 0.25,
            'solubility': 0.20
        }
    
    def calculate_risk(self, predictions):
        """
        Calculate overall development risk score.
        
        Args:
            predictions: dict containing all ADMET predictions
            
        Returns:
            dict: Risk score and breakdown
        """
        try:
            risks = {}
            
            # Toxicity risk (ClinTox probability)
            clintox_data = predictions.get('toxicity', {}).get('clintox', {})
            if isinstance(clintox_data, dict) and clintox_data.get('probability') is not None:
                risks['toxicity'] = clintox_data['probability']
            else:
                risks['toxicity'] = 0.5  # Unknown = moderate risk
            
            # Permeability risk (inverse of BBB probability)
            perm_data = predictions.get('permeability', {})
            bbb_data = perm_data.get('bbb', {}) if isinstance(perm_data, dict) else {}
            
            if isinstance(bbb_data, dict) and bbb_data.get('probability') is not None:
                risks['permeability'] = 1 - bbb_data['probability']
            else:
                risks['permeability'] = 0.5
            
            # Drug-likeness risk (inverse of score/100)
            dl_data = predictions.get('druglikeness', {})
            if isinstance(dl_data, dict) and dl_data.get('score') is not None:
                risks['druglikeness'] = 1 - (dl_data['score'] / 100.0)
            else:
                risks['druglikeness'] = 0.5
            
            # Solubility risk (based on logS)
            props = predictions.get('properties', {})
            logs = props.get('logS') if isinstance(props, dict) else None
            
            if logs is not None and isinstance(logs, (int, float)):
                # logS < -6: very poor, -6 to -4: poor, -4 to -2: moderate, > -2: good
                if logs < -6:
                    risks['solubility'] = 0.9
                elif logs < -4:
                    risks['solubility'] = 0.6
                elif logs < -2:
                    risks['solubility'] = 0.3
                else:
                    risks['solubility'] = 0.1
            else:
                risks['solubility'] = 0.5
            
            # Calculate weighted overall risk
            overall_risk = sum(risks[k] * self.weights[k] for k in risks.keys())
            overall_risk = max(0, min(1, overall_risk))  # Clamp to [0, 1]
            
            # Convert to 0-100 scale
            risk_score = overall_risk * 100
            
            # Determine risk level
            if risk_score < 30:
                risk_level = "Low"
                risk_color = "green"
            elif risk_score < 60:
                risk_level = "Moderate"
                risk_color = "orange"
            else:
                risk_level = "High"
                risk_color = "red"
            
            return {
                'overall_score': float(risk_score),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'components': {
                    'toxicity': float(risks['toxicity'] * 100),
                    'permeability': float(risks['permeability'] * 100),
                    'druglikeness': float(risks['druglikeness'] * 100),
                    'solubility': float(risks['solubility'] * 100)
                },
                'weights': self.weights
            }
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            return {
                'overall_score': None,
                'risk_level': 'Unknown',
                'risk_color': 'gray',
                'components': {},
                'weights': self.weights
            }

if __name__ == "__main__":
    # Test
    engine = RiskEngine()
    
    # Mock predictions
    test_predictions = {
        'toxicity': {
            'clintox': {'probability': 0.15}
        },
        'permeability': {
            'bbb': {'probability': 0.8}
        },
        'druglikeness': {
            'score': 75
        },
        'properties': {
            'logS': -3.5
        }
    }
    
    result = engine.calculate_risk(test_predictions)
    print(f"Risk Score: {result['overall_score']:.1f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Components: {result['components']}")
