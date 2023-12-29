import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph
import pickle

import warnings
# Filter out the UserWarning from StellarGraph
warnings.filterwarnings("ignore", category=UserWarning, module="stellargraph")

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import sparse_categorical_crossentropy  # Use sparse categorical cross-entropy
import tensorflow as tf
from sklearn import model_selection
from tensorflow.keras.models import load_model
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from stellargraph.mapper import PaddedGraphGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, optimizers, losses, metrics, models
from tensorflow.keras.callbacks import EarlyStopping

# Load data from JSON files
data_dir = 'C:\\Users\\Saqib\\PycharmProjects\\GAGE\\output_AED'

data = []
labels = []

max_nodes = 128  # Maximum number of nodes for each graph

for class_folder in os.listdir(data_dir):
    print(class_folder)
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
num_classes = len(set(labels))  # Calculate the number of unique classes
class_to_index = {c: i for i, c in enumerate(set(labels))}
labels = [class_to_index[label] for label in labels]

# Create a StellarGraph object for each graph in your dataset

# Define a list to store individual graph data
graphs = []

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
        G.add_edge(source, target, Edge_Features=decimal_fe)  # Add edge features
        i += 1

    # Create a StellarGraph from the NetworkX graph
    stellar_graph = sg.StellarGraph.from_networkx(G, node_features="features", edge_type_attr="Edge_Features")

    graphs.append(stellar_graph)

# Split data into train, validation, and test sets
train_data, test_data, train_labels, test_labels = train_test_split(graphs, labels, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Generator
generator = PaddedGraphGenerator(graphs=train_data)
# Save the generator to a file
with open('models/padded_graph_generator_20231009_GCN2_v1_acc_enhance_v3_gamarue_included.pkl', 'wb') as file:
    pickle.dump(generator, file)

k = 35
layer_sizes = [64, 32, 16, num_classes]  # Adjust the last layer size for multi-class

dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "softmax"],  # Use softmax for multi-class
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=64, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)

predictions = Dense(units=num_classes, activation="softmax")(x_out)  # Use softmax for multi-class

model = Model(inputs=x_inp, outputs=predictions)

model.compile(
    optimizer=Adam(lr=0.0001), loss=sparse_categorical_crossentropy, metrics=["acc"],
)

train_gen = generator.flow(train_data, train_labels, batch_size=50, symmetric_normalization=False)
val_gen = generator.flow(val_data, val_labels, batch_size=1, symmetric_normalization=False)
test_gen = generator.flow(test_data, test_labels, batch_size=1, symmetric_normalization=False)

early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

history = model.fit(
    train_gen, epochs=100, verbose=1, validation_data=val_gen, shuffle=True, callbacks=[early_stopping]
)
# Save the trained model
# Define a function to save the model
def save_model(model, model_name):
    model.save(model_name)
model_name = "models/GCN_20231009_GCN2_v1_acc_enhance_v3_gamarue_included.h5"
save_model(model, model_name)
sg.utils.plot_history(history)
plt.show()
# Evaluate the model on the test set
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# Make predictions on the test set
Y_pred = model.predict(test_gen)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Classification report
class_names = [index for index, _ in sorted(class_to_index.items(), key=lambda x: x[1])]
report = classification_report(test_labels, Y_pred_classes, target_names=class_names)
print(report)

# Confusion matrix
confusion = confusion_matrix(test_labels, Y_pred_classes)
print(confusion)
print(class_to_index)
