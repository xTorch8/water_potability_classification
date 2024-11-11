"""Correlation Plot Module

This module provides a function to display a heatmap of correlations between
columns in a DataFrame along with the correlation values.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_plot(df, save_path = "../src/visualization/plot/correlation_plot.png"):
    """
    Displays a heatmap of correlation values between numerical columns in a DataFrame.

    Parameters:
    - df (DataFrame): The pandas DataFrame for which to compute and plot correlations.
    - save_path (str): File path to save the plot image.

    Returns:
    - None

    Prints:
    - Correlation values between numerical columns.

    Raises:
    ValueError: If the DataFrame does not contain numerical columns for correlation.
    TypeError: If the input is not a pandas DataFrame.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data is not a valid pandas DataFrame")
        
        correlation_matrix = df.corr()     

        if correlation_matrix.empty:
            raise ValueError("DataFrame does not contain numerical columns for correlation.")

        print("Correlation Values:")
        print(correlation_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm", fmt = ".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")

        plt.savefig(save_path)
        plt.show()
    except TypeError as type_error:
        print(f"[ERR] {type_error}")
    except ValueError as value_error:
        print(f"[ERR] {value_error}")
    except Exception as error:
        print(f"[ERR] An unexpected exception occurred: {error}")