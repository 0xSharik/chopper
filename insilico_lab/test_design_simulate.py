
"""
test_design_simulate.py — Verification of the Design & Simulate Workflow.
"""
import unittest
import os
import shutil
from src.workflows.design_and_simulate import DesignAndSimulate
from src.workflows.validation import validate_smiles

class TestDesignWorkflow(unittest.TestCase):
    
    def setUp(self):
        # Setup temporary output dir
        self.test_dir = "data/test_workflow"
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.workflow = DesignAndSimulate()
        
        # Aspirin
        self.parent_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        
        # Fast config for testing
        self.config = {
            "mode": "demo",
            "output_base_dir": self.test_dir,
            "production_ns": 0.002, 
            "equilibration_nvt_ps": 1.0,
            "equilibration_npt_ps": 1.0, # Will trigger warning/skip in engine but run fast
            "energy_minimization_steps": 100
        }

    def test_validation(self):
        """Test structure validation."""
        self.assertTrue(validate_smiles("CCCCCC")) # Hexane - OK (6 atoms > 5)
        
        with self.assertRaises(ValueError):
             validate_smiles("C") # Too small (1 atom)
             
        with self.assertRaises(ValueError):
             validate_smiles("C.C") # Disconnected

    def test_workflow_execution(self):
        """Test full modify -> simulate pipeline."""
        print("\n🧪 Testing Modify & Simulate Workflow...")
        
        try:
            # 1. Add Methyl
            result = self.workflow.modify_and_simulate(
                parent_smiles=self.parent_smiles,
                modification_type="Add Methyl",
                md_config=self.config,
                molecule_label="test_aspirin"
            )
            
            # Check structure
            self.assertNotEqual(result["parent_smiles"], result["modified_smiles"])
            self.assertTrue(len(result["modified_smiles"]) > len(self.parent_smiles))
            
            # Check metrics exist
            metrics = result["md_metrics"]
            self.assertIn("rmsd_mean", metrics)
            self.assertIn("stability_score", metrics)
            
            print(f"✅ Workflow Succeeded! New SMILES: {result['modified_smiles']}")
            print(f"📊 New Stability Score: {metrics['stability_score']}")
            
        except Exception as e:
            self.fail(f"Workflow failed with error: {e}")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

if __name__ == "__main__":
    unittest.main()
