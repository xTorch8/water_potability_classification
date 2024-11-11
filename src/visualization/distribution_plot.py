"""Distribution Visualization Module

This module provides functions to visualize distribution in a specified numerical column
of a DataFrame using histograms, box plots, and scatter plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def distribution_plot(df, column_name, save_path = "../src/visualization/plot/distribution_plot.png"):
    """
    Visualizes distribution in a specified numerical column using a histogram, box plot, and scatter plot.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - column_name (str): The column name for which to visualize outliers.
    - save_path (str): File path to save the plot image.

    Returns:
    - None

    Raises:
    - ValueError: If the specified column does not exist in the DataFrame or is not numerical.
    - TypeError: If the input is not a pandas DataFrame.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data is not a valid pandas DataFrame")
        
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise ValueError(f"Column '{column_name}' is not numerical.")

        plt.figure(figsize = (18, 5))

        # Histogram
        plt.subplot(1, 3, 1)
        sns.histplot(df[column_name], kde = True, bins = 30, color = "skyblue")
        plt.title(f'Histogram of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel("Frequency")

        # Box Plot
        plt.subplot(1, 3, 2)
        sns.boxplot(x = df[column_name], color = "lightcoral")
        plt.title(f'Box Plot of {column_name}')
        plt.xlabel(column_name)

        # Scatter Plot
        plt.subplot(1, 3, 3)
        plt.scatter(df.index, df[column_name], alpha = 0.6, color="mediumseagreen")
        plt.title(f'Scatter Plot of {column_name}')
        plt.xlabel('Index')
        plt.ylabel(column_name)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
    except TypeError as type_error:
        print(f"[ERROR] {type_error}")
    except ValueError as value_error:
        print(f"[ERROR] {value_error}")
    except Exception as error:
        print(f"[ERROR] An unexpected exception occurred: {error}")
