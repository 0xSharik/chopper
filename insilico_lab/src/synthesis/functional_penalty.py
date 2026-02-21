from rdkit import Chem

class FunctionalGroupPenalty:
    """Calculates penalties for reactive or undesirable functional groups."""
    
    def __init__(self):
        # Patterns and associated penalty weights
        self.patterns = {
            "Nitro": ("[$([NX3](=O)=O),$([NX3+]([O-])=O)]", 2.0),
            "Peroxide": ("OO", 3.0),
            "Azide": ("N=[N+]=[N-]", 4.0),
            "Geminal dihalide": ("[CX4]([Cl,Br,I])([Cl,Br,I])", 2.0),
            "Quaternary C": ("[CX4]([#6])([#6])([#6])([#6])", 1.0)
        }
        self.smarts = {name: Chem.MolFromSmarts(smarts) for name, (smarts, weight) in self.patterns.items()}

    def score(self, mol) -> float:
        """Score based on presence of tricky functional groups (0-10)."""
        total_penalty = 0.0
        
        for name, pattern in self.smarts.items():
            if pattern and mol.HasSubstructMatch(pattern):
                weight = self.patterns[name][1]
                matches = len(mol.GetSubstructMatches(pattern))
                total_penalty += weight * matches
                
        # Scale to 0-10
        return round(min(total_penalty, 10.0), 2)
