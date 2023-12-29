import os
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt


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
directory_path_gage = r'C:\Users\Saqib\PycharmProjects\GAGE\robustness_files\graph_features\all_gage_cfg_files'
directory_path_cfg = r'C:\Users\Saqib\PycharmProjects\GAGE\robustness_files\graph_features\all_cfgexplainer_files'

# Get a list of CSV files in the directory
csv_files_gage = [os.path.join(directory_path_gage, filename) for filename in os.listdir(directory_path_gage) if
             filename.endswith('.csv')]
csv_files_cfg = [os.path.join(directory_path_cfg, filename) for filename in os.listdir(directory_path_cfg) if
             filename.endswith('.csv')]

# Load the CSV files into a list of DataFrames (one DataFrame per class)
class_dataframes_gage = [pd.read_csv(csv_file) for csv_file in csv_files_gage]
class_dataframes_cfg = [pd.read_csv(csv_file) for csv_file in csv_files_cfg]
# Generate all permutations of class indices
class_indices = list(range(len(class_dataframes_gage)))
class_permutations = list(itertools.combinations(class_indices, 2))

# Initialize lists to store MMD values for each number of rows considered
num_rows_to_consider_values = list(range(1, 6))  # 1 to 5 rows
mmd_scores_gage = []
mmd_scores_cfg = []

def plot_robustness_graph(class1, class2):
    # Calculate MMD for each number of rows considered
    for num_rows_to_consider in num_rows_to_consider_values:
        # Select the first num_rows_to_consider rows from each class
        class1_data_cfg = class_dataframes_cfg[class1].iloc[:num_rows_to_consider, 63:]
        class2_data_cfg = class_dataframes_cfg[class2].iloc[:num_rows_to_consider, 63:]

        class1_data_gage = class_dataframes_gage[class1].iloc[:num_rows_to_consider]
        class2_data_gage = class_dataframes_gage[class2].iloc[:num_rows_to_consider]

        # Assuming your data is in a format that MMD expects (numpy arrays or similar)
        # You may need to scale or preprocess your data accordingly
        class1_data_np_gage = class1_data_gage.values
        class2_data_np_gage = class2_data_gage.values

        class1_data_np_cfg = class1_data_cfg.values
        class2_data_np_cfg = class2_data_cfg.values

        # Calculate MMD using the compute_mmd function
        sigma = 1.0  # You can adjust the bandwidth parameter as needed
        mmd_value_cfg = compute_mmd(class1_data_np_cfg, class2_data_np_cfg, sigma)
        mmd_value_gage = compute_mmd(class1_data_np_gage, class2_data_np_gage, sigma)

        # Append the MMD value to the list
        mmd_scores_cfg.append(mmd_value_cfg)
        mmd_scores_gage.append(mmd_value_gage)

    # Create a plot
    plt.figure(figsize=(8, 6))
    plt.plot(num_rows_to_consider_values, mmd_scores_cfg, marker='o', linestyle='-', color='green', label='CFG Data')
    plt.plot(num_rows_to_consider_values, mmd_scores_gage, marker='o', linestyle='-', color='red', label='GAGE Data')
    plt.xlabel('Data size')
    plt.ylabel('MMD Score')
    # plt.title('MMD between Class ' + str(class1) + ' and Class ' + str(class2))
    class1_name = "Firseria"
    class2_name = "Gamarue"
    plt.title('MMD between class ' + class1_name + ' and class ' + class2_name)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()
    print('MMD between class ' + class1_name + ' and class ' + class2_name)
    print("CFGExplainer : ", mmd_scores_cfg)
    print("GAGE : ", mmd_scores_gage)
plot_robustness_graph(5, 6)

