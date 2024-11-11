"""Data Cleaning Module

This module provides functions to clean a dataset by handling missing values, duplicates, 
invalid data types, and remove outliers. It also ensures columns are appropriately formatted 
for analysis.
"""
import numpy as np
import pandas as pd

def remove_duplicates(df):
    """
    Removes duplicate rows from the DataFrame.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame to clean.
    
    Returns:
    - DataFrame: The cleaned DataFrame without duplicate rows.
    """
    try:
        df_cleaned = df.drop_duplicates()
        print(f"Removed {df.shape[0] - df_cleaned.shape[0]} duplicate rows.")
        return df_cleaned
    except Exception as e:
        print(f"[ERR] Failed to remove duplicates: {e}")
        return df

def handle_missing_values(df, strategy = "drop", fill_value = None):
    """
    Handles missing values in the DataFrame by either dropping or filling them.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame to clean.
    - strategy (str): The strategy for handling missing values. Options are:
        - "drop" (default): Drops rows with missing values.
        - "fill": Fills missing values with the provided fill_value.
    - fill_value: The value to use for filling missing data (used only if strategy is "fill").
    
    Returns:
    DataFrame: The cleaned DataFrame with handled missing values.
    """
    try:
        if strategy == "drop":
            df_cleaned = df.dropna()
            print(f"Dropped {df.isnull().sum().sum()} missing values.")
        elif strategy == "fill":
            if fill_value is None:
                df_cleaned = df.interpolate()
                # raise ValueError("fill_value must be specified when strategy is 'fill'")
            else:
                df_cleaned = df.fillna(fill_value)
            print(f"Filled missing values with {fill_value}.")
        else:
            raise ValueError("Invalid strategy. Choose either 'drop' or 'fill'.")
        return df_cleaned
    except ValueError as e:
        print(f"[ERR] Value error: {e}")
        return df
    except Exception as e:
        print(f"[ERR] Failed to handle missing values: {e}")
        return df

def check_data_types(df, expected_types):
    """
    Checks if the columns in the DataFrame match the expected data types.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame to check.
    - expected_types (dict): A dictionary where keys are column names and values are the expected data types.
    
    Returns:
    None. Prints the status of data types in the DataFrame.
    """
    try:
        for column, expected_type in expected_types.items():
            actual_type = df[column].dtype
            if actual_type != expected_type:
                print(f"[WARN] Column '{column}' has type {actual_type}, expected {expected_type}.")
            else:
                print(f"Column '{column}' is correctly of type {expected_type}.")
    except KeyError as e:
        print(f"[ERR] Column '{e.args[0]}' not found in DataFrame.")
    except Exception as e:
        print(f"[ERR] Error checking data types: {e}")

def remove_outliers(df, column, distribution_type):
    """
    Removes or handles outliers in a specified column of a DataFrame based on the distribution type.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column in which to identify and handle outliers.
    - distribution_type (str): The type of distribution. Options are 'normal', 'skewed', 'uniform', and 'polynomial'.

    Returns:
    - pd.DataFrame: A DataFrame with outliers handled as specified, or the original DataFrame if an error occurs.

    Raises:
    - ValueError: If an invalid distribution type is provided.
    """
    try:
        if distribution_type == "normal":
            mean = df[column].mean()
            std_dev = df[column].std()

            df = df[(df[column] >= mean - 3 * std_dev) & (df[column] <= mean + 3 * std_dev)]
        elif distribution_type == "skewed":
            Q1 = df[column].quantile(0.25)  
            Q3 = df[column].quantile(0.75)  
            IQR = Q3 - Q1  

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        elif distribution_type == "uniform":
            lower_limit = df[column].quantile(0.05)
            upper_limit = df[column].quantile(0.95)

            df[column] = np.clip(df[column], lower_limit, upper_limit)

        elif distribution_type == "polynomial":
            df = df[df[column] > 0]

            df[column] = np.log(df[column])
        else:
            raise ValueError("Invalid strategy. Choose either 'normal', 'skewed', 'uniform', or 'polynomial'.")
        
        return df
    except Exception as e:
        # Handle any exceptions and print an error message
        print(f"[ERR] Failed to remove outliers: {e}")
        return df

def remove_empty_columns(df):
    """
    Removes columns that are completely empty from the DataFrame.
    
    Parameters:
    df (DataFrame): The pandas DataFrame to clean.
    
    Returns:
    DataFrame: The cleaned DataFrame with empty columns removed.
    """
    try:
        df_cleaned = df.dropna(axis = 1, how = "all")
        print(f"Removed {df.shape[1] - df_cleaned.shape[1]} empty columns.")
        return df_cleaned
    except Exception as e:
        print(f"[ERR] Failed to remove empty columns: {e}")
        return df

def clean_data(df, missing_strategy = "drop", fill_value = None, expected_types = None, distribution_dict = None):
    """
    Performs a complete data cleaning pipeline: removes duplicates, handles missing values,
    checks data types, and removes empty columns.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame to clean.
    - missing_strategy (str): The strategy for handling missing values ("drop" or "fill").
    - fill_value (any): The value to fill missing values (used if strategy is "fill").
    - expected_types (dict): Dictionary specifying expected column types for validation.
    
    Returns:
    - DataFrame: The cleaned DataFrame.
    """
    try:
        print("Starting data cleaning process...")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input is not Pandas DataFrame")
        
        # Step 1: Remove duplicates
        df = remove_duplicates(df)
        
        # Step 2: Handle missing values
        df = handle_missing_values(df, strategy = missing_strategy, fill_value = fill_value)
        
        # Step 3: Check data types
        if expected_types is not None:
            check_data_types(df, expected_types)
        
        # Step 4: Remove outliers
        if distribution_dict is not None:
            for column in distribution_dict:
                df = remove_outliers(df, column, distribution_dict[column])

        # Step 5: Remove empty columns
        df = remove_empty_columns(df)
        
        print("Data cleaning process completed.")
        return df
    except Exception as e:
        print(f"[ERR] Unexpected error during data cleaning: {e}")
        return df
