
import openmm as mm

print("Available OpenMM Platforms:")
for i in range(mm.Platform.getNumPlatforms()):
    p = mm.Platform.getPlatform(i)
    print(f"  - {p.getName()}")

print("\nDefault Platform:")
try:
    # Quick dummy system to test default selection
    system = mm.System()
    system.addParticle(1.0)
    integrator = mm.VerletIntegrator(0.002)
    ctx = mm.Context(system, integrator, p)
    print(f"  - {ctx.getPlatform().getName()}")
    
    if p.getName() == "OpenCL":
        try:
             # Try to get device info
             print(f"    OpenCL Device Index: {p.getPropertyValue(ctx, 'OpenCLDeviceIndex')}")
             # OpenMM OpenCL often doesn't expose name easily without context property?
             # platform.getPropertyValue(context, property)
             # Property 'OpenCLPlatformIndex' or similar?
             # Actually 'OpenCLDeviceName' might actully be a property?
             # Let's try to just list all properties?
        except:
             pass
except Exception as e:
    print(f"  Error checking default: {e}")

print("\nPlugin Load Failures:")
errors = mm.Platform.getPluginLoadFailures()
if not errors:
    print("  None")
else:
    for e in errors:
        print(f"  - {e}")

print("\nDetailed Platform Info:")
speed = 0.0
try:
    # Benchmark to force device initialization
    # And try to get device name from Context
    p = mm.Platform.getPlatformByName("OpenCL")
    system = mm.System()
    system.addParticle(1.0)
    integrator = mm.VerletIntegrator(0.002)
    # create context
    ctx = mm.Context(system, integrator, p)
    
    # Try different property keys known for OpenCL
    keys = ["OpenCLDeviceName", "DeviceName", "OpenCLPlatformName"]
    found = False
    for k in keys:
        try:
            val = p.getPropertyValue(ctx, k)
            print(f"  OpenCL Property '{k}': {val}")
            found = True
        except:
             pass
    
    if not found:
        print("  Could not retrieve specific OpenCL Device Name property.")

except Exception as e:
    print(f"  Could not inspect OpenCL details: {e}")
