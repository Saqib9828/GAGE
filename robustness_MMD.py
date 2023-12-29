import os
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def compute_mmd(x, y, sigma=1.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of data.

    Parameters:
    - x, y: arrays of shape (n_samples, n_features) representing two sets of data.
    - sigma: float, the bandwidth parameter of the RBF kernel.

    Returns:
    - float, the MMD between the two sets of data.
    """
    kxx = rbf_kernel(x, x, gamma=1.0 / (2.0 * sigma ** 2))
    kxy = rbf_kernel(x, y, gamma=1.0 / (2.0 * sigma ** 2))
    kyy = rbf_kernel(y, y, gamma=1.0 / (2.0 * sigma ** 2))
    mmd = np.mean(kxx) - 2.0 * np.mean(kxy) + np.mean(kyy)
    return mmd


# Define the directory path
directory_path = r'C:\Users\Saqib\PycharmProjects\GAGE\robustness_files\graph_features\all_gage_cfg_files'

# Get a list of CSV files in the directory
csv_files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if
             filename.endswith('.csv')]

# Load the CSV files into a list of DataFrames (one DataFrame per class)
class_dataframes = [pd.read_csv(csv_file) for csv_file in csv_files]

# Define the number of rows to consider
num_rows_to_consider = 5  # You can change this to any number of rows you want to consider

# Generate all permutations of class indices
class_indices = list(range(len(class_dataframes)))
class_permutations = list(itertools.combinations(class_indices, 2))

# Initialize an array to store MMD values for each permutation
mmd_values = []

# Calculate MMD for each permutation
for class_permutation in class_permutations:
    class1_index, class2_index = class_permutation
    class1_data = class_dataframes[class1_index].iloc[:num_rows_to_consider]
    class2_data = class_dataframes[class2_index].iloc[:num_rows_to_consider]

    # Assuming your data is in a format that MMD expects (numpy arrays or similar)
    # You may need to scale or preprocess your data accordingly
    class1_data_np = class1_data.values
    class2_data_np = class2_data.values

    # Calculate MMD using the compute_mmd function
    sigma = 1.0  # You can adjust the bandwidth parameter as needed
    mmd_value = compute_mmd(class1_data_np, class2_data_np, sigma)

    # Append the MMD value to the array
    mmd_values.append(mmd_value)

# Print MMD values for all permutations
for idx, mmd_value in enumerate(mmd_values):
    class1_index, class2_index = class_permutations[idx]
    print(f'MMD between class {class1_index} and class {class2_index}: {mmd_value}')
