import pandas as pd
import matplotlib.pyplot as plt
import os

# Load Data
data_path = "data/md_runs_verify/test_aspirin/state_data.csv"
output_dir = "data/md_runs_verify/test_aspirin/plots"
os.makedirs(output_dir, exist_ok=True)

try:
    df = pd.read_csv(data_path)
    # Clean columns (strip whitespace/quotes)
    df.columns = [c.strip().replace('"', '') for c in df.columns]
    
    # 1. Potential Energy
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (ps)'], df['Potential Energy (kJ/mole)'], color='blue', alpha=0.7)
    plt.title("Potential Energy vs Time")
    plt.xlabel("Time (ps)")
    plt.ylabel("Potential Energy (kJ/mol)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "potential_energy.png"))
    plt.close()
    
    # 2. Temperature
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (ps)'], df['Temperature (K)'], color='red', alpha=0.7)
    plt.axhline(y=300, color='black', linestyle='--', label="Target 300K")
    plt.title("Temperature vs Time")
    plt.xlabel("Time (ps)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "temperature.png"))
    plt.close()
    
    # 3. Density (if varies)
    if 'Density (g/mL)' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Time (ps)'], df['Density (g/mL)'], color='green', alpha=0.7)
        plt.title("Density vs Time")
        plt.xlabel("Time (ps)")
        plt.ylabel("Density (g/mL)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "density.png"))
        plt.close()
        
    print(f"Plots generated in {output_dir}")

except Exception as e:
    print(f"Failed to generate plots: {e}")
