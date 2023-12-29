import os
import json
import numpy as np
import networkx as nx
import stellargraph as sg
from stellargraph.layer import DeepGraphCNN
import pickle

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
import pandas as pd
import pickle

# Load data from JSON files
data_dir = 'C:\\Users\\Saqib\\PycharmProjects\\GAGE\\output_AED_try_graphclass'
graph_gen = 'models/padded_graph_generator_20231007.pkl'
model_name = "models/GCN_20231009.h5"

# Predict classes for all the loaded data
def predict_classes(graphs, graph_labels, loaded_gen, model_name):
    graph_pred_data = loaded_gen.flow(
        list(range(len(graphs))),  # Use all graphs
        targets=graph_labels,
        batch_size=1,
        symmetric_normalization=False,
    )

    custom_objects = {
        "GraphConvolution": sg.layer.GraphConvolution,
        "SortPooling": sg.layer.SortPooling
    }
    model_loaded = tf.keras.models.load_model(model_name, custom_objects=custom_objects)
    predictions = model_loaded.predict(graph_pred_data)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

if __name__ == '__main__':
    data = []
    labels = []

    max_nodes = 128  # Maximum number of nodes for each graph

    for class_folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, class_folder)):
            class_label = class_folder
            class_folder_path = os.path.join(data_dir, class_folder)
            for json_file in os.listdir(class_folder_path):
                if json_file.endswith('.json'):
                    with open(os.path.join(class_folder_path, json_file), 'r') as file:
                        graph_data = json.load(file)

                        # Check if JSON data has nodes and edges
                        if "Nodes" not in graph_data or "Edges" not in graph_data:
                            print(f"JSON file {json_file} has no nodes or edges, skipping...")
                            continue

                        # Extract only CPID values from nodes
                        nodes = [node["CPID"] for node in graph_data["Nodes"]]

                        if not nodes:
                            print(f"JSON file {json_file} has no node data, skipping...")
                            continue

                        # Limit the number of nodes to max_nodes
                        nodes = nodes[:max_nodes]

                        # Filter edges connected to nodes within the first max_nodes
                        edges = [edge for edge in graph_data["Edges"] if
                                 edge["Source"] in nodes or edge["Destination"] in nodes]
                        # Extract edge features
                        edge_features = []
                        for edge in edges:
                            edge_features.append([
                                edge.get("is_consequent", False),
                                edge.get("is_conditional", False),
                                edge.get("intra_edge", False),
                                edge.get("external_edge", False)
                            ])
                        # Extract node features based on CPID values
                        node_features = {node["CPID"]: node["AED_features"] for node in graph_data["Nodes"] if
                                         node["CPID"] in nodes}

                        graph_data["Nodes"] = nodes
                        graph_data["Edges"] = edges
                        graph_data["NodeFeatures"] = node_features
                        graph_data["EdgeFeatures"] = edge_features

                        data.append(graph_data)
                        labels.append(class_label)

    # Encode class labels
    num_classes = len(set(labels))
    class_to_index = {c: i for i, c in enumerate(set(labels))}
    labels = [class_to_index[label] for label in labels]

    # Create a StellarGraph object for each graph in your dataset

    # Define a list to store individual graph data
    graphs = []
    graph_labels = labels

    # Iterate through your data and create a StellarGraph for each graph
    for graph_data in data:
        # Create an empty graph
        G = nx.Graph()

        # Add nodes to the graph
        for node_id in graph_data["Nodes"]:
            G.add_node(node_id, features=np.array(graph_data["NodeFeatures"][node_id][0]))

        # Add edges to the graph
        i = 0
        for edge in graph_data["Edges"]:
            source, target = edge["Source"], edge["Destination"]
            fe = graph_data["EdgeFeatures"][i]
            int_fe = int_list = [int(x) for x in fe]
            decimal_fe = sum(b << i for i, b in enumerate(reversed(int_fe)))
            G.add_edge(source, target, Edge_Features=decimal_fe)
            i += 1

        # Create a StellarGraph from the NetworkX graph
        stellar_graph = sg.StellarGraph.from_networkx(G, node_features="features", edge_type_attr="Edge_Features")

        graphs.append(stellar_graph)

    with open(graph_gen, 'rb') as file:
        loaded_gen = pickle.load(file)

    predicted_classes = predict_classes(graphs, graph_labels, loaded_gen, model_name)
    print(predicted_classes)
