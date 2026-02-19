
"""
verify_validation.py — Headless test for Validation Module
"""
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.validation.benchmark_logp import LogPBenchmark

def test_benchmark():
    print("🚀 Testing LogP Benchmark...")
    
    try:
        bench = LogPBenchmark()
        results = bench.run_benchmark()
        
        print("\n✅ Benchmark execution successful!")
        print(f"   MAE:  {results['mae']:.3f}")
        print(f"   RMSE: {results['rmse']:.3f}")
        
        # Verify files
        csv_path = results['csv_path']
        plot_path = os.path.join("artifacts", "benchmark_logp.png")
        
        if os.path.exists(csv_path):
            print(f"   ✅ CSV exported: {csv_path}")
        else:
            print("   ❌ CSV missing!")
            
        if os.path.exists(plot_path):
            print(f"   ✅ Plot generated: {plot_path}")
        else:
            print("   ❌ Plot missing!")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_benchmark()
