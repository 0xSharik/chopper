import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import logging

# Add src to path if needed (though running as module is preferred)
current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, '../..'))

from src.utils.logging_config import setup_logging

# Setup logger
setup_logging()
logger = logging.getLogger(__name__)

def generate_qc_report(data_path, output_dir):
    """
    Generate QC report from cleaned data.
    
    Args:
        data_path (str): Path to cleaned CSV.
        output_dir (str): Directory to save QC artifacts.
    """
    logger.info(f"Generating QC report for {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Failed to load data for QC: {e}")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Statistics
    stats = {
        "count": int(len(df)),
        "mean_solubility": float(df['solubility'].mean()),
        "std_solubility": float(df['solubility'].std()),
        "min_solubility": float(df['solubility'].min()),
        "max_solubility": float(df['solubility'].max())
    }
    
    # Save statistics
    stats_path = os.path.join(output_dir, "esol_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Statistics saved to {stats_path}")
    print(json.dumps(stats, indent=4))

    # 2. Validation Checks
    print("\n--- Validation Checks ---")
    if -12 <= stats['min_solubility'] and stats['max_solubility'] <= 2:
        print("[PASS] Solubility range is within expected bounds (-12 to +2).")
    else:
        print(f"[WARNING] Solubility range ({stats['min_solubility']} to {stats['max_solubility']}) outside typical bounds.")
        
    if -4.5 <= stats['mean_solubility'] <= -2.5: # broadened slightly around -3 to -4
        print(f"[PASS] Mean solubility {stats['mean_solubility']:.2f} is within expected range.")
    else:
        print(f"[WARNING] Mean solubility {stats['mean_solubility']:.2f} is outside expected range (-3 to -4).")

    # Critical failure checks
    if stats['max_solubility'] > 20 or stats['min_solubility'] < -50:
        logger.critical("Data corruption detected: Extreme solubility values found!")
        print("[CRITICAL] Data corruption detected: Extreme solubility values found!")
    
    # 3. Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df['solubility'], kde=True, bins=30)
    plt.title('Distribution of Aqueous Solubility (LogS)')
    plt.xlabel('Log Solubility (mol/L)')
    plt.ylabel('Count')
    plt.axvline(stats['mean_solubility'], color='r', linestyle='--', label=f"Mean: {stats['mean_solubility']:.2f}")
    plt.legend()
    
    plot_path = os.path.join(output_dir, "esol_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Distribution plot saved to {plot_path}")
    print(f"Distribution plot saved to {plot_path}")

if __name__ == "__main__":
    # Define paths based on project structure
    # Assumption: script run from project root or src/data_pipeline
    # Adjusting for likely execution from project root
    
    # Try to find the data file relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    
    input_file = os.path.join(project_root, 'data/processed/esol_cleaned.csv')
    output_dir = os.path.join(project_root, 'data/metadata')
    
    if not os.path.exists(input_file):
        # Fallback for execution from different CWD
        input_file = 'data/processed/esol_cleaned.csv'
        output_dir = 'data/metadata'

    generate_qc_report(input_file, output_dir)
