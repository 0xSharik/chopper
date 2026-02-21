import json
from rdkit import Chem
import logging

logger = logging.getLogger(__name__)

class MixtureParser:
    """Parses various multi-molecule input formats."""
    
    def parse(self, input_data) -> list[dict]:
        """
        Parse input data into a standardized list of SMILES and counts.
        
        Supported formats:
        1. List of dicts: [{"smiles": "...", "count": 5}, ...]
        2. Comma-separated: "SMILES1, SMILES2, ..."
        3. Dot-separated: "SMILES1.SMILES2. ..."
        """
        if isinstance(input_data, list):
            return self._parse_json_style(input_data)
        elif isinstance(input_data, str):
            if input_data.strip().startswith('[') or input_data.strip().startswith('{'):
                try:
                    data = json.loads(input_data)
                    return self._parse_json_style(data)
                except json.JSONDecodeError:
                    pass
            return self._parse_string_style(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _parse_json_style(self, data) -> list[dict]:
        if not isinstance(data, list):
            data = [data]
        
        results = []
        for item in data:
            if not isinstance(item, dict) or 'smiles' not in item:
                continue
            smiles = item['smiles'].strip()
            count = int(item.get('count', 1))
            if self._validate_smiles(smiles):
                results.append({"smiles": smiles, "count": count})
        return results

    def _parse_string_style(self, input_str) -> list[dict]:
        # Handle dot-separated (common in chemistry) or comma-separated
        input_str = input_str.replace(',', '.')
        parts = [p.strip() for p in input_str.split('.') if p.strip()]
        
        # Count occurrences of each SMILES
        counts = {}
        for p in parts:
            if self._validate_smiles(p):
                counts[p] = counts.get(p, 0) + 1
                
        return [{"smiles": s, "count": c} for s, c in counts.items()]

    def _validate_smiles(self, smiles) -> bool:
        if not smiles:
            return False
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES in mixture: {smiles}")
            return False
        return True
