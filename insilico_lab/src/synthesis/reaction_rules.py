from rdkit import Chem
from rdkit.Chem import AllChem
from .utils import validate_smiles, canonicalize

class RetrosynthesisRules:
    """Applies single-step retrosynthesis rules."""
    
    def __init__(self):
        # Define structured reaction rules
        self.reaction_defs = {
            "Ester cleavage": {
                "smarts": "[#6:1](=O)O[#6:2]>>[#6:1](=O)O.[#6:2]O",
                "type": "Hydrolysis",
                "confidence": 0.95
            },
            "Amide cleavage": {
                "smarts": "[#6:1](=O)N[#6:2]>>[#6:1](=O)O.[#6:2]N",
                "type": "Amidolysis",
                "confidence": 0.95
            },
            "Alcohol oxidation": {
                "smarts": "[#6&!$(C=O):1][OH1]>>[#6:1]=O",
                "type": "Oxidation",
                "confidence": 0.5
            },
            "Halogen removal": {
                "smarts": "[#6:1]-[Cl,Br,I]>>[#6:1]",
                "type": "Dehalogenation",
                "confidence": 0.6
            },
            "Alkyl halide elimination": {
                "smarts": "[#6&H:1]-[#6]([Cl,Br,I])>>[#6:1]=[#6]",
                "type": "Elimination",
                "confidence": 0.4
            }
        }
        self.reactions = {
            name: {
                "rxn": AllChem.ReactionFromSmarts(rd["smarts"]),
                "type": rd["type"],
                "confidence": rd["confidence"]
            }
            for name, rd in self.reaction_defs.items()
        }

    def generate_precursors(self, mol):
        """Generate single-step precursors with structured metadata."""
        precursors = []
        
        for name, data in self.reactions.items():
            rxn = data["rxn"]
            try:
                products = rxn.RunReactants((mol,))
                for prod_set in products:
                    smiles_parts = []
                    valid_set = True
                    for p in prod_set:
                        try:
                            Chem.SanitizeMol(p)
                            smiles_parts.append(Chem.MolToSmiles(p, canonical=True))
                        except:
                            valid_set = False
                            break
                    
                    if valid_set:
                        combined_smiles = ".".join(sorted(list(set(smiles_parts))))
                        precursors.append({
                            "smiles": combined_smiles,
                            "reaction_type": data["type"],
                            "confidence": data["confidence"],
                            "name": name
                        })
            except Exception:
                continue
                
        # Sort by confidence descending and de-duplicate by smiles
        seen_smiles = set()
        unique_precursors = []
        for p in sorted(precursors, key=lambda x: x["confidence"], reverse=True):
            if p["smiles"] not in seen_smiles:
                unique_precursors.append(p)
                seen_smiles.add(p["smiles"])
                
        return unique_precursors[:5]
