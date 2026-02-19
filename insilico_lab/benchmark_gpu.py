
"""
benchmark_gpu.py — Benchmark OpenMM performance on the current system.
"""
import time
import openmm
import openmm.app as app
import openmm.unit as unit
from src.md.config import DEFAULT_MD_CONFIG

def benchmark():
    print("🚀 Running GPU Benchmark...")
    
    # 1. Setup Dummy System (DHFR-like size or just Aspirin?)
    # Let's use Aspirin from SMILES to be realistic to user's case
    from src.md.system_builder import SystemBuilder
    
    config = DEFAULT_MD_CONFIG.copy()
    config['mode'] = 'research' # force good parameters
    
    print("   Building System (Aspirin + Solvent)...")
    builder = SystemBuilder(config)
    topology, system, positions = builder.build_system("CC(=O)Oc1ccccc1C(=O)O", "benchmark_mol")
    
    print(f"   System Size: {system.getNumParticles()} atoms")
    
    # 2. Setup Simulation
    dt_fs = 2.0
    integrator = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, dt_fs*unit.femtoseconds)
    
    # Try getting OpenCL
    try:
        platform = openmm.Platform.getPlatformByName("OpenCL")
        props = {"OpenCLPrecision": "mixed"}
        print(f"   Platform: OpenCL (mixed precision)")
    except:
        platform = openmm.Platform.getPlatformByName("CPU")
        props = {}
        print(f"   Platform: CPU (fallback)")
        
    simulation = app.Simulation(topology, system, integrator, platform, props)
    simulation.context.setPositions(positions)
    
    # Minimize briefly
    print("   Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=100)
    
    # 3. Benchmark
    steps = 5000
    print(f"   Running {steps} steps...")
    
    start = time.time()
    simulation.step(steps)
    end = time.time()
    
    elapsed = end - start
    ns = (steps * dt_fs) / 1e6
    speed_ns_day = (ns / elapsed) * 86400
    
    print(f"\n🏁 Benchmark Results:")
    print(f"   Time: {elapsed:.2f} s")
    print(f"   Speed: {speed_ns_day:.1f} ns/day")
    
    if speed_ns_day > 100:
        print("   ✅ GPU is performing well!")
    else:
        print("   ⚠️ Performance seems low (or CPU is being used).")

if __name__ == "__main__":
    benchmark()
