"""
test_md_pipeline.py — Phase 1 MD infrastructure integration test.

Tests the full pipeline on Aspirin (SMILES: CC(=O)Oc1ccccc1C(=O)O).

Uses a short 10 ps production run (production_ns=0.01, output_frame_ps=0.1)
for fast CI-friendly verification. For a full 5 ns production run, use
the default MolecularDynamicsEngine() without config override.

Assertions:
  1. Output directory created: data/md_runs/aspirin/
  2. Metrics dict returned with all required keys
  3. rmsd_mean is not NaN
  4. energy_var is positive (> 0)

Run:
    python test_md_pipeline.py
"""

import os
import sys
import math

# Make sure project root is on path
_test_dir = os.path.dirname(os.path.abspath(__file__))
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

from src.md import MolecularDynamicsEngine

# -------------------------------------------------------------------------
# Test parameters
# -------------------------------------------------------------------------
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
MOLECULE_ID    = "aspirin"

# Override for fast test: 10 ps production (instead of 5 ns)
TEST_CONFIG = {
    "production_ns":        0.01,   # 10 ps production
    "output_frame_ps":      0.1,    # frame every 100 fs → 100 frames
    "equilibration_nvt_ps": 2.0,    # 2 ps NVT (fast)
    "equilibration_npt_ps": 4.0,    # 4 ps NPT (fast)
    "energy_minimization_steps": 500,
    "seed": 42,
}

REQUIRED_METRIC_KEYS = {
    "rmsd_mean", "rmsd_std", "rg_mean",
    "sasa_mean", "hbond_avg", "energy_var",
}


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def check(condition: bool, name: str, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


# -------------------------------------------------------------------------
# Run test
# -------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Chopper — MD Pipeline Integration Test")
    print(f"  Molecule: {MOLECULE_ID}")
    print(f"  SMILES:   {ASPIRIN_SMILES}")
    print(f"  Config:   fast ({TEST_CONFIG['production_ns']*1000:.0f} ps production)")
    print("=" * 60)

    engine = MolecularDynamicsEngine(config=TEST_CONFIG)
    result = engine.run(smiles=ASPIRIN_SMILES, molecule_id=MOLECULE_ID)

    print("\nPipeline result:")
    print(f"  status: {result.get('status')}")
    if result.get("status") == "error":
        print(f"  error:  {result.get('message')}")

    print("\nAssertions:")
    results = []

    # 1. Output directory exists
    expected_dir = os.path.join(_test_dir, "data", "md_runs", MOLECULE_ID)
    results.append(check(
        os.path.isdir(expected_dir),
        "Output directory created",
        expected_dir,
    ))

    # 2. Status is success
    results.append(check(
        result.get("status") == "success",
        "Pipeline status == 'success'",
        result.get("message", ""),
    ))

    metrics = result.get("metrics", {})

    # 3. All required metric keys present
    missing = REQUIRED_METRIC_KEYS - set(metrics.keys())
    results.append(check(
        len(missing) == 0,
        "All metric keys present",
        f"missing={missing}" if missing else "OK",
    ))

    # 4. rmsd_mean is not NaN
    rmsd_mean = metrics.get("rmsd_mean", float("nan"))
    results.append(check(
        not math.isnan(rmsd_mean),
        f"rmsd_mean is not NaN",
        f"rmsd_mean={rmsd_mean}",
    ))

    # 5. energy_var is positive
    energy_var = metrics.get("energy_var", 0.0)
    results.append(check(
        energy_var > 0,
        "energy_var > 0",
        f"energy_var={energy_var}",
    ))

    # 6. trajectory.dcd exists
    traj_path = os.path.join(expected_dir, "trajectory.dcd")
    results.append(check(
        os.path.isfile(traj_path),
        "trajectory.dcd exists",
        traj_path,
    ))

    # 7. metadata.json exists
    meta_path = os.path.join(expected_dir, "metadata.json")
    results.append(check(
        os.path.isfile(meta_path),
        "metadata.json exists",
        meta_path,
    ))

    # 8. md.log exists
    log_path = os.path.join(expected_dir, "md.log")
    results.append(check(
        os.path.isfile(log_path),
        "md.log exists",
        log_path,
    ))

    # Summary
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} assertions passed")
    print("='*60")

    if passed == total:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗ — check md.log for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
