import pandas as pd
import os
import sys

def load_data(filepath):
    """
    Load data from a given filepath.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise IOError(f"Error reading file: {e}")

    # Rename columns to match requirements if necessary
    if 'measured log solubility in mols per litre' in df.columns:
        df = df.rename(columns={'measured log solubility in mols per litre': 'solubility'})

    required_columns = ['smiles', 'solubility']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
        
    print(f"Dataset loaded. Shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    # Example usage mostly for testing
    try:
        # Assuming script is run from project root or src/data_pipeline
        # Adjust path as necessary based on execution context
        # Try relative path from this file location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, '../../data/raw/esol.csv')
        df = load_data(data_path)
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")
