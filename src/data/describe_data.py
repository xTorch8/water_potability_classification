"""
Data Description Module

This module provides a function to display a summary of key information
about a dataset, including a preview, shape, summary statistics, data types,
missing values, and duplicate counts.
"""

import pandas as pd

def describe_data(df):
    """
    Displays a comprehensive summary of a DataFrame's key characteristics.

    Parameters:
    df (DataFrame): The pandas DataFrame to describe.

    Returns:
    None

    Prints:
    - A preview of the first 5 rows of data.
    - Shape of the data (rows, columns).
    - Summary statistics for numeric columns.
    - Data types of each column.
    - Count of missing values in each column.
    - Count of duplicate rows in the dataset.

    Raises:
    TypeError: If the input is not a pandas DataFrame.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input is not a valid Pandas DataFrame")
             
        print("Data Preview: ")
        print(df.head(5))

        print("Data Shape: ")
        print(df.shape)

        print("Summary Statistics: ")
        print(df.describe())

        print("Data Types: ")
        print(df.dtypes)

        print("Missing Data: ")
        print(df.isnull().sum())

        print("Duplicates Data: ")
        print(df.duplicated().sum())
    except TypeError as type_error:
        print(f"[ERR] {type_error}")