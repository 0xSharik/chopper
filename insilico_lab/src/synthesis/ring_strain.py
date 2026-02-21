from rdkit import Chem

class RingStrainAnalyzer:
    """Analyzes ring strain based on heuristics."""
    
    def score(self, mol) -> float:
        """Calculate ring strain score (0-10)."""
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        
        total_strain = 0.0
        
        # Ring size penalties
        for ring in rings:
            size = len(ring)
            if size == 3:
                total_strain += 3.0
            elif size == 4:
                total_strain += 2.0
            elif size == 5:
                total_strain += 0.5
            elif size == 6:
                total_strain += 0.0
            elif size >= 7:
                total_strain += 2.0
                
        # Bridgehead and Spiro atoms
        # Bridgehead: atom in more than 2 rings (simplified)
        # Spiro: atom sharing exactly 2 rings but with more than 3 bonds in rings
        atom_rings = [0] * mol.GetNumAtoms()
        for ring in rings:
            for atom_idx in ring:
                atom_rings[atom_idx] += 1
                
        for idx, count in enumerate(atom_rings):
            if count > 2:
                total_strain += 1.0 # Bridgehead/Multi-ring junction
            elif count == 2:
                # Check if spiro (all bonds to that atom are in rings)
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetDegree() == 4:
                    total_strain += 1.0 # Potential Spiro
                    
        # Cap at 10.0
        return round(min(total_strain, 10.0), 2)
