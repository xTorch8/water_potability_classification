"""
Data Loading Module

This module provides a function to load data from a specified file path.
It reads CSV files into a pandas DataFrame and handles potential errors. 
"""
import pandas as pd

def load_data(file_path):
    """
    Loads a dataset from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file to be loaded.

    Returns:
    - DataFrame: A pandas DataFrame containing the loaded data, or None if an error occurs.

    Raises:
    - FileNotFoundError: If the file at the specified path does not exist.
    - pd.errors.EmptyDataError: If the file is empty.
    - pd.errors.ParserError: If there is a parsing issue with the file.    
    """
    try:
        df = pd.read_csv(file_path)

        print(f"Successfully loaded data from {file_path}")

        return df
    except FileNotFoundError:
        print(f"[ERR] The file path at {file_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print("[ERR] The file is empty.")
        return
    except pd.errors.ParserError:
        print("[ERR] There was a parsing issue with the file")
        return