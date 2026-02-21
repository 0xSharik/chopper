from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import logging
import os
import pandas as pd
from .utils import validate_smiles
from .sas import SyntheticAccessibility
from .ring_strain import RingStrainAnalyzer
from .functional_penalty import FunctionalGroupPenalty
from ..modeling.applicability_domain import ApplicabilityDomain
from ..inference.property_engine import PropertyEngine

logger = logging.getLogger(__name__)

class ForwardSynthesisEngine:
    """Controlled Forward Coupling Engine for precise chemical transformations."""
    
    def __init__(self):
        # 1. Initialize scoring engines
        self.sas_engine = SyntheticAccessibility()
        self.ring_engine = RingStrainAnalyzer()
        self.functional_engine = FunctionalGroupPenalty()
        self.prop_engine = PropertyEngine() # For ADMET previews
        self.morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
        
        # 1.1 Initialize Applicability Domain
        self.ad_engine = ApplicabilityDomain()
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            data_path = os.path.join(project_root, 'data', 'processed', 'lipophilicity_features.csv')
            
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, nrows=1000)
                self.ad_engine.fit(df['smiles'].tolist())
            else:
                logger.warning(f"AD training data not found at {data_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Applicability Domain: {e}")

        # 2. Define exactly TWO forward reactions with structured metadata
        # Confidence: 0.9+ -> High, 0.7-0.9 -> Moderate, <0.7 -> Low
        self.reaction_defs = {
            "Ester formation": {
                "smarts": "[CX3:1](=O)[OX2H1].[OX2H:2][CX4:3] >> [CX3:1](=O)[OX2:2][CX4:3]",
                "roles": [
                    Chem.MolFromSmarts("[CX3](=O)[OX2H1]"), # Acid
                    Chem.MolFromSmarts("[OX2H][CX4]")      # Alcohol
                ],
                "confidence_score": 0.95,
                "type": "Coupling"
            },
            "Amide formation": {
                "smarts": "[CX3:1](=O)[OX2H1].[NX3:2][CX4:3] >> [CX3:1](=O)[NX3:2][CX4:3]",
                "roles": [
                    Chem.MolFromSmarts("[CX3](=O)[OX2H1]"), # Acid
                    Chem.MolFromSmarts("[NX3;H2,H1][CX4]")  # Amine
                ],
                "confidence_score": 0.95,
                "type": "Coupling"
            }
        }
        
        self.reactions = {}
        for name, d in self.reaction_defs.items():
            rxn = AllChem.ReactionFromSmarts(d["smarts"])
            errors, _ = rxn.Validate()
            if errors > 0:
                logger.error(f"Reaction {name} SMARTS is invalid")
            self.reactions[name] = rxn

    def synthesize(self, precursor_smiles_list: list) -> dict:
        """
        Generate virtual products with strict role matching and validation.
        """
        if len(precursor_smiles_list) != 2:
            return {"products": []}

        # Deterministic sorting
        sorted_smiles = sorted(set(precursor_smiles_list))
        precursors = []
        for s in sorted_smiles:
            try:
                mol = validate_smiles(s)
                if mol:
                    precursors.append(mol)
            except Exception as e:
                logger.warning(f"Invalid precursor skipped: {s} ({e})")

        if len(precursors) != 2:
            return {"products": []}

        raw_products = []
        
        # 3. Enforce Reactant Role Matching
        # Reactions are sorted to ensure deterministic processing order
        for name in sorted(self.reactions.keys()):
            rxn = self.reactions[name]
            roles = self.reaction_defs[name]["roles"]
            
            # Try both permutations for role matching
            for p1, p2 in [(0, 1), (1, 0)]:
                mol1 = precursors[p1]
                mol2 = precursors[p2]
                
                if mol1.HasSubstructMatch(roles[0]) and mol2.HasSubstructMatch(roles[1]):
                    try:
                        results = rxn.RunReactants((mol1, mol2))
                        for prod_set in results:
                            for prod in prod_set:
                                prod_data = self._process_and_score_product(prod, precursors, name)
                                if prod_data:
                                    raw_products.append(prod_data)
                    except Exception as e:
                        logger.error(f"Error applying {name}: {e}")

        # 6. Remove Duplicates using canonical SMILES
        unique_products = {}
        for p in raw_products:
            if p["smiles"] not in unique_products:
                unique_products[p["smiles"]] = p
        
        # 7. Rank by plausibility_score (ascending, lower is better)
        # Sort by score then SMILES for absolute determinism
        # Formula: plausibility = (0.5 * SAS + 0.2 * ring + 0.2 * functional + 0.1 * (1-reaction_confidence))
        sorted_products = sorted(
            unique_products.values(), 
            key=lambda x: (x["plausibility_score"], x["smiles"])
        )
        
        # Return top 2 candidates
        return {"products": sorted_products[:2]}

    def _process_and_score_product(self, mol, precursors, reaction_name) -> dict:
        """Fully validate, sanitize, and score a synthetic product."""
        try:
            # 5. Sanity Filter
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            
            # No radicals
            if any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms()):
                return None
            
            # Disconnected fragments
            if len(Chem.GetMolFrags(mol)) > 1:
                return None
            
            # Formal charge imbalance (must be neutral for common coupling products)
            if Chem.GetFormalCharge(mol) != 0:
                return None
            
            # Nitro group limit (max 3)
            nitro_smarts = Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+]([O-])=O)]")
            if len(mol.GetSubstructMatches(nitro_smarts)) > 3:
                return None
                
            # Heavy atom limit
            if mol.GetNumHeavyAtoms() > 80:
                return None

            # 4. Mass Conservation
            # Water loss (O is 1 heavy atom) or similar leaving groups
            total_input_atoms = sum(p.GetNumHeavyAtoms() for p in precursors)
            product_atoms = mol.GetNumHeavyAtoms()
            if abs(product_atoms - total_input_atoms) > 2:
                return None

            # 6. Reject if product is identical to a precursor (no self-reproduction)
            precursor_smiles = [Chem.MolToSmiles(p, canonical=True) for p in precursors]
            if smiles in precursor_smiles:
                return None

            # 7. ADMET Screening (Ro5)
            props = self.prop_engine.predict_properties(smiles)
            mw = props.get('MW', 0)
            logp = props.get('logP', 0)
            hbd = props.get('HBD', 0)
            hba = props.get('HBA', 0)
            
            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if hbd > 5: violations += 1
            if hba > 10: violations += 1
            
            # Reject automatically if > 2 violations
            if violations > 2: return None

            # 8. Scoring
            sas = self.sas_engine.calculate(mol)
            ring = self.ring_engine.score(mol)
            functional = self.functional_engine.score(mol)
            
            rxn_data = self.reaction_defs[reaction_name]
            rxn_conf = rxn_data["confidence_score"]
            
            # New Ranking Formula: 0.5*SAS + 0.2*Ring + 0.2*Functional + 0.1*(1-reaction_confidence)
            plausibility = (0.5 * sas) + (0.2 * ring) + (0.2 * functional) + (0.1 * (1 - rxn_conf))
            
            # 9. Quality Classification
            if plausibility < 15:
                p_class = "High Synthetic Plausibility"
            elif plausibility < 35:
                p_class = "Moderate Plausibility"
            else:
                p_class = "Low Plausibility"

            # 10. Real Applicability Domain Check
            ad_res = self.ad_engine.predict([smiles])[0]
            is_in_domain = ad_res['in_domain']
            similarity = ad_res['max_similarity']

            return {
                "smiles": smiles,
                "reaction_name": reaction_name,
                "reaction_confidence": "High" if rxn_conf >= 0.9 else "Moderate" if rxn_conf >= 0.7 else "Low",
                "sas": sas,
                "ring": ring,
                "functional": functional,
                "plausibility_score": round(plausibility, 3),
                "classification": p_class,
                "admet_preview": {
                    "MW": round(mw, 2),
                    "logP": round(logp, 2),
                    "HBD": hbd,
                    "HBA": hba,
                    "Ro5_status": "Pass" if violations <= 1 else "Borderline" if violations == 2 else "Fail"
                },
                "in_domain": is_in_domain,
                "domain_similarity": round(float(similarity), 3),
                "confidence": "High" if is_in_domain else "Low confidence synthetic plausibility"
            }

        except Exception as e:
            logger.debug(f"Product validation failed: {e}")
            return None
