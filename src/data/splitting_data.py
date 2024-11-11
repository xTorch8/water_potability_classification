"""
Splitting Data Module

This module provides a function to split data into training, 
validation, and testing dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def splitting_data(df, X, y):
    """
    Split dataset into training, validation, and testing datasets.

    Parameters:
    - df (DataFrame): The full DataFrame containing the dataset.
    - X (DataFrame): DataFrame of features (input variables).
    - y (Series): Series of labels (target variable).

    Returns:
    - list: A list containing three DataFrames for training, validation, and testing datasets.

    Raises:
    - TypeError: If any of df, X, or y are not Pandas DataFrames.
    - Exception: If any other error occurs during data splitting.
    """

    try:
        if not isinstance(df, pd.DataFrame) or not isinstance(X, pd.DataFrame):
            raise TypeError("df and/or x is not a valid Pandas DataFrame")
        
        if not isinstance(y, pd.Series):
            raise TypeError("y is not a valid Pandas Series")
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, stratify = y_temp, random_state = 42)

        train_dataset = pd.concat([X_train, y_train], axis = 1)
        validation_dataset = pd.concat([X_val, y_val], axis = 1)
        testing_dataset = pd.concat([X_test, y_test], axis = 1)

        return [train_dataset, validation_dataset, testing_dataset]
    except Exception as e:
        print(f"[ERR] An error occured when splitting data: {e}")
        return None