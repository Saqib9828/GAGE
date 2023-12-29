import os
import json
import csv
import numpy as np
import networkx as nx

# Path to the 'nodes' and 'edges' folders
nodes_folder = 'C:\\Users\\Saqib\\PycharmProjects\\GAGE\\robustness_files\\benign\\gage_cfg\\nodes'
edges_folder = 'C:\\Users\\Saqib\\PycharmProjects\\GAGE\\robustness_files\\benign\\gage_cfg\\edges'
csv_file_path = "C:\\Users\\Saqib\\PycharmProjects\\GAGE\\robustness_files\\benign\\gage_cfg\\graph_features_benign.csv"


# Function to create a graph from node and edge data
def create_graph(node_data, edge_data):
    G = nx.Graph()

    # Add nodes with attributes
    for node_id, features in node_data.items():
        G.add_node(node_id, features=features)

    # Add edges with attributes
    for edge in edge_data:
        source = edge['Source']
        destination = edge['Destination']
        G.add_edge(source, destination, attributes=edge)

    return G

# Function to calculate graph features
def compute_graph_features(graph):
    features = []

    # Get all node features into a list of arrays
    all_node_features = []
    for node_id, data in graph.nodes(data=True):
        if 'features' in data:
            node_features = data['features']
            all_node_features.append(node_features)

    # Calculate the mean of node features vertically
    if all_node_features:
        mean_node_features = np.mean(all_node_features, axis=0).tolist()
        features.extend(mean_node_features)
    else:
        features.extend([0.0] * 64)  # or any default value if no nodes have 'features'

    # Other features
    features.append(len(list(graph.edges())))
    features.append(nx.number_of_selfloops(graph))
    features.append(min(dict(graph.degree()).values()))
    features.append(min(dict(graph.degree(weight='weight')).values()))  # Updated for in-degree
    features.append(min(dict(graph.degree(weight='weight')).values()))  # Updated for out-degree
    features.append(np.mean(list(dict(graph.degree()).values())))
    features.append(np.mean(list(dict(graph.degree(weight='weight')).values())))  # Updated for in-degree
    features.append(np.mean(list(dict(graph.degree(weight='weight')).values())))  # Updated for out-degree
    features.append(max(dict(graph.degree()).values()))
    features.append(max(dict(graph.degree(weight='weight')).values()))  # Updated for in-degree
    features.append(max(dict(graph.degree(weight='weight')).values()))  # Updated for out-degree

    # Add more features as needed (e.g., subtree, graphlets, walks)

    return features


# Iterate through the files in 'nodes' and 'edges' folders
all_graphs_features = []
for nodes_filename, edges_filename in zip(os.listdir(nodes_folder), os.listdir(edges_folder)):
    if nodes_filename.endswith('.json') and edges_filename.endswith('.json'):
        nodes_path = os.path.join(nodes_folder, nodes_filename)
        edges_path = os.path.join(edges_folder, edges_filename)

        # Load node and edge data from JSON files
        with open(nodes_path, 'r') as nodes_file, open(edges_path, 'r') as edges_file:
            node_data = json.load(nodes_file)["AED_features"]
            edge_data = json.load(edges_file)

        # Create a NetworkX graph
        graph = create_graph(node_data, edge_data)
        # Calculate graph features
        graph_features = compute_graph_features(graph)

        # Print or store the features as needed
        print(f"Graph {nodes_filename} features: {len(graph_features)}")
        print("--------------------------")
        # Flatten the nested list
        flat_list = []

        # Flatten the nested list
        for item in graph_features:
            if isinstance(item, list):
                flat_list.extend(item)
            else:
                flat_list.append(item)
        all_graphs_features.append(flat_list)

# Specify the CSV file path

# Save the flat list to a CSV file
with open(csv_file_path, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    for flat_list in all_graphs_features:
        csvwriter.writerow(flat_list)
print("Data saved to graph_features.csv")


