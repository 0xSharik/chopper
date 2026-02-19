
"""
stress_test_gpu.py — High-load simulation to force GPU visibility.
"""
import time
import openmm
import openmm.app as app
import openmm.unit as unit

def stress_test():
    print("🔥 Starting GPU Stress Test...")
    print("   Goal: Generate sustained load to verify Task Manager usage.")
    
    # 1. Create a Large Box of Water (Heavy System)
    # 5x5x5 nm box of water is ~4000 waters => ~12000 atoms
    print("   Building large water box (approx 12,000 atoms)...")
    
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(app.ForceField('tip3p.xml'), boxSize=openmm.Vec3(5,5,5)*unit.nanometers)
    
    topology = modeller.getTopology()
    positions = modeller.getPositions()
    
    system = app.ForceField('tip3p.xml').createSystem(
        topology, 
        nonbondedMethod=app.PME, 
        nonbondedCutoff=1.0*unit.nanometers, 
        constraints=app.HBonds
    )
    
    print(f"   System built: {system.getNumParticles()} atoms")
    
    # 2. Select OpenCL on GPU
    try:
        platform = openmm.Platform.getPlatformByName("OpenCL")
        # create context safely
        ctx_temp = openmm.Context(system, openmm.VerletIntegrator(0.002), platform)
        dev_name = platform.getPropertyValue(ctx_temp, "OpenCLDeviceName")
        print(f"   Target Device: {dev_name}") 
        del ctx_temp
    except Exception as e:
        print(f"   Error identifying platform: {e}")
        platform = openmm.Platform.getPlatformByName("OpenCL")

    # 3. integrator
    integrator = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
    
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    
    print("   Minimizing (puts heavy load on GPU)...")
    simulation.minimizeEnergy(maxIterations=100)
    
    # 4. Long Run
    steps = 50000 # 100 ps
    print(f"   Running {steps} steps (approx 10-20 seconds of load)...")
    print("   👉 PLEASE CHECK TASK MANAGER -> PERFORMANCE -> GPU 0 -> CHECK 'CUDA' or 'Compute_0' GRAPH NOW!")
    
    start = time.time()
    simulation.step(steps)
    end = time.time()
    
    elapsed = end - start
    ns_day = (steps * 0.002 * 1e-6 / elapsed) * 86400
    
    print(f"\n✅ Stress Test Complete.")
    print(f"   Time: {elapsed:.2f} s")
    print(f"   Performance: {ns_day:.1f} ns/day")

if __name__ == "__main__":
    try:
        stress_test()
    except KeyboardInterrupt:
        print("\nStopped.")
