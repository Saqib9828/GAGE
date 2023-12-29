import random
import math
import matplotlib.pyplot as plt
import warnings
# Filter out the UserWarning from StellarGraph
warnings.filterwarnings("ignore", category=UserWarning, module="stellargraph")

#-----------------
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
data_dir = 'C:\\Users\\Saqib\\PycharmProjects\\GAGE\\output\\CEG_GA_try'
json_file_path = "C:\\Users\\Saqib\\PycharmProjects\\GAGE\output\\CEG_GA_try\\malware1\\origin_449__.tmp0.json" #firseria
extracted_ex_file_path = "output/extracted_ex__1_firseria_origin_449__.tmp0.json"
graph_gen = 'models/padded_graph_generator_20231007.pkl'
model_name = "models/GCN_20231009.h5"
with open(graph_gen, 'rb') as file:
    loaded_gen = pickle.load(file)
custom_objects = {
        "GraphConvolution": sg.layer.GraphConvolution,
        "SortPooling": sg.layer.SortPooling
    }
loaded_model = tf.keras.models.load_model(model_name, custom_objects=custom_objects)

def plot_common_nodes_and_edges2(g1, g2):
    """
    Plot a StellarGraph object with common nodes and edges highlighted from another StellarGraph.

    Args:
        g1 (stellargraph.StellarGraph): The first StellarGraph.
        g2 (stellargraph.StellarGraph): The second StellarGraph (subgraph of g1).

    Returns:
        None
    """
    # Convert StellarGraphs to NetworkX graphs
    nx_g1 = g1.to_networkx()
    nx_g2 = g2.to_networkx()

    # Extracting nodes and edges that are common in both graphs
    common_nodes = set(nx_g1.nodes()).intersection(set(nx_g2.nodes()))
    common_edges = set(nx_g1.edges()).intersection(set(nx_g2.edges()))
    plt.figure(figsize=(10, 10))
    # Creating a layout for our nodes
    layout = nx.kamada_kawai_layout(nx_g1)

    # Drawing nodes and edges of g1
    nx.draw(nx_g1, pos=layout, with_labels=False, node_color='black', edge_color='black', node_size=50)
    # Example: Color nodes differently if they connect to more than one community

    #colors = ['red', 'blue', 'green', 'yellow', 'purple']
    #color_map = [random.choice(colors) for node in common_nodes]
    # Drawing common nodes and edges with different color and labels
    nx.draw_networkx_nodes(nx_g1, pos=layout, nodelist=common_nodes, node_color='red', node_size=150)
    nx.draw_networkx_edges(nx_g1, pos=layout, edgelist=common_edges, edge_color='red', width=2)
    nx.draw_networkx_labels(nx_g1, pos=layout, labels={node: node for node in common_nodes}, font_color='black', font_size=10)

    # Adding a title and displaying the plot
    plt.title("Graph g1 with common nodes and edges highlighted")
    plt.show()




def remove_isolated_nodes(graph):
    nx_graph = graph.to_networkx()

    # Find the nodes that have at least one neighbor (connected to an edge)
    connected_nodes = [node for node, degree in nx_graph.degree if degree > 0]

    # Create a new StellarGraph with only the connected nodes
    subgraph = graph.subgraph(connected_nodes)

    return subgraph

def JSONtoGraphData(json_file_path):
    # class_label = "Default"
    max_nodes = 128  # Maximum number of nodes for each graph


    # Load the JSON data from the file
    with open(json_file_path, 'r') as file:
        graph_data = json.load(file)

        # Check if JSON data has nodes and edges
        if "Nodes" not in graph_data or "Edges" not in graph_data:
            print(f"JSON file {json_file_path} has no nodes or edges, skipping...")
        else:
            # Extract only CPID values from nodes
            nodes = [node["CPID"] for node in graph_data["Nodes"]]

            if not nodes:
                print(f"JSON file {json_file_path} has no node data, skipping...")
            else:
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

    return graph_data

def graphDataToNXGraph(graph_data):
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

    return stellar_graph


# Predict classes for all the loaded data
def predict_classes(graphs, graph_labels, loaded_gen, model_loaded):
    graph_pred_data = loaded_gen.flow(
        list(range(len(graphs))),  # Use all graphs
        targets=graph_labels,
        batch_size=1,
        symmetric_normalization=False,
    )

    predictions = model_loaded.predict(graph_pred_data)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes[0], predictions[0]

def create_chromosome(length, edges):
    return [random.randint(0, len(edges)-1) for i in range(length)]

def create_population(size, length, edges):
    # create a population as a list of random chromosomes
    return [create_chromosome(length, edges) for i in range(size)]

def encode_edges(graph):
    # Create a mapping between edge indices and edges
    edge_mapping = {}
    edges = list(graph['Edges'])
    for i, edge in enumerate(edges):
        edge_mapping[i] = edge
    return edge_mapping

def convert_chromosome_format(chromosome):
    # Check if the chromosome is a list of integers
    if isinstance(chromosome, list) and all(isinstance(x, int) for x in chromosome):
        # If it's already in the desired format, return it as is
        return chromosome
    elif isinstance(chromosome, list) and all(isinstance(x, int) for x in chromosome[0]):
        # If it's in the format of a list within a list, convert it to the desired format
        return chromosome[0]
    else:
        # If the format is not recognized, raise an error or handle it as needed
        raise ValueError("Invalid chromosome format")

def decode_chromosome(chromosome, edge_mapping):
    print(chromosome)
    chromosome = convert_chromosome_format(chromosome)
    new_edges = []
    for index in chromosome:
        edge = edge_mapping[index]
        new_edges.append(edge)
    return new_edges

def decoded_cromoToGraph(graph_2_edges, graph_1):
    Edges_features_temp = []
    for i in range(len(graph_2_edges)):
        Edges_features_temp.append(
            [graph_2_edges[i]['is_consequent'], graph_2_edges[i]['is_conditional'], graph_2_edges[i]['intra_edge'],
             graph_2_edges[i]['external_edge']])
    g_temp = {'Nodes': graph_1['Nodes'], 'Edges': graph_2_edges, 'NodeFeatures': graph_1['NodeFeatures'],
              'EdgeFeatures': Edges_features_temp}
    return graphDataToNXGraph(g_temp)

def fitness(sub_graph, parent_health):
    _, child_health = predict_classes([sub_graph], graph_labels, loaded_gen, loaded_model)
    squared_distances = [(child_health[i] - parent_health[i])**2 for i in range(len(child_health))]
    distance = math.sqrt(sum(squared_distances))
    return distance

def get_fittest(population, n_top, edges_map, parent_health):
    fitness_list = []
    for cromo in population:
        decoded_cromo = decode_chromosome(cromo, edges_map)
        nx_graph = decoded_cromoToGraph(decoded_cromo, parent_graph)
        nx_reduced_graph = remove_isolated_nodes(nx_graph)
        fitness_list.append(fitness(nx_reduced_graph, parent_health))
    indices = list(range(len(fitness_list)))
    sorted_indices = sorted(indices, key=lambda i: fitness_list[i])
    n_top_index = sorted_indices[:n_top]
    fittest_cromo = [population[i] for i in n_top_index]
    return fittest_cromo

def crossover(a, b):
    # randomly select a crossover point
    index = random.randint(0, len(a)-1)
    # create a new chromosome by combining the first part of a with the second part of b
    child = a[:index] + b[index:]
    # return the new chromosome
    return child

def mutation(chromosome, rate):
    # randomly select some indices to mutate
    indices = [i for i in range(len(chromosome)) if random.random() < rate]
    # mutate the selected indices by adding a random integer between -1 and 1
    for i in indices:
        chromosome[i] += random.randint(-1, 1)
    # return the mutated chromosome
    return chromosome

# define the genetic algorithm function
def genetic_algorithm(population_size, cromo_length, parent_edges, edges_map, parent_graph, parent_health, mutation_rate, n_top, generations=10):
    # create an initial population of chromosomes
    # population = create_population(population_size, chromosome_length)
    population = create_population(population_size, cromo_length, parent_edges) #initial population
    # print(population)
    # iterate over the specified number of generations
    gererations_fitness = []
    for i in range(generations):
        # print("Generation:", i,  " | ", generations)
        # create a new population by selecting and breeding the fittest individuals
        new_population = []
        for j in range(population_size):
            print("Generation: ", i, " | ", generations, " Population: ", j, " | ", population_size)
            # select two parent chromosomes using tournament selection
            fittest_cromo = get_fittest(population, n_top, edges_map, parent_health)
            child = crossover(fittest_cromo[0], fittest_cromo[1])
            child = mutation(child, mutation_rate)

            # add the child chromosome to the new population
            new_population.append(child)
        # replace the old population with the new population
        population = new_population
        top_fit_cromo = get_fittest(population, 1, edges_map, parent_health)
        top_fit_edges = decode_chromosome(top_fit_cromo, edges_map)
        final_subgraph = decoded_cromoToGraph(top_fit_edges, parent_graph)
        gererations_fitness.append(fitness(final_subgraph, parent_health))
    # return the fittest chromosome in the final population
    return get_fittest(population, 1, edges_map, parent_health), gererations_fitness

if __name__ == '__main__':
    # -----------------------
    cromo_length = 25
    population_size = 8
    n_top = 2 # number of fittest cromo extract from popu
    mutation_rate = 0.2
    generations = 10
    # ----------------------
    graph_labels = ['Default']
    parent_graph = JSONtoGraphData(json_file_path)
    nx_parent_graph = graphDataToNXGraph(parent_graph)
    parent_edges = nx_parent_graph.edges()
    predicted_class, parent_health = predict_classes([nx_parent_graph], graph_labels, loaded_gen, loaded_model)
    edges_map = encode_edges(parent_graph)

    # fittest_in_generations, generations_fitness = genetic_algorithm(population_size, cromo_length, parent_edges, edges_map, parent_graph, parent_health, mutation_rate, n_top, generations)
    # print(fittest_in_generations, "\n", generations_fitness)
    fittest_in_generations = [201, 82, 195, 89, 94, 174, 32, 210, 216, 111, 164, 53, 165, 92, 126, 188, 208, 128, 48, 97, 48, 172, 203, 106, 130]
    fittest_decoded_cromo = decode_chromosome(fittest_in_generations, edges_map)
    fittest_nx_graph = decoded_cromoToGraph(fittest_decoded_cromo, parent_graph)
    fittest_nx_reduced_graph = remove_isolated_nodes(fittest_nx_graph)
    print(fittest_nx_reduced_graph.info())
    predicted_class, prediction_prob = predict_classes([fittest_nx_reduced_graph], graph_labels, loaded_gen, loaded_model)
    print("predicted_class:", predicted_class)
    print("prediction_prob:", prediction_prob)
    # plot_graphs_with_highlight(nx_parent_graph, fittest_nx_reduced_graph)
    # Specify the file path where you want to save the JSON file

    # Write the list of dictionaries to a JSON file
    with open(extracted_ex_file_path, "w") as json_file:
        json.dump(fittest_decoded_cromo, json_file, indent=4)

    print("Data saved to", extracted_ex_file_path)
    #plot_graph_with_highlight_diff(nx_parent_graph, fittest_nx_reduced_graph)
    plot_common_nodes_and_edges2(nx_parent_graph, fittest_nx_reduced_graph)


    # cromo = create_chromosome(cromo_length, parent_edges)
    #
    # population = create_population(population_size, cromo_length, parent_edges)
    # fittest_cromo = get_fittest(population, n_top, edges_map, parent_health)
    # child = crossover(fittest_cromo[0], fittest_cromo[1])
    # child = mutation(child, mutation_rate)
    # print(child)

    # decoded_cromo = decode_chromosome(cromo, edges_map)
    # nx_graph_2 = decoded_cromoToGraph(decoded_cromo, parent_graph)
    # print(nx_parent_graph.info())
    # print(nx_graph_2.info())
    # nx_reduced_graph_2 = remove_isolated_nodes(nx_graph_2)
    # print(nx_iso_graph_2.info())
    # fitness_child = fitness(nx_iso_graph_2, parent_health)
    # predicted_class = predict_classes([nx_graph_2], graph_labels, loaded_gen, loaded_model)
    # print(fitness_child)
    # plot_graphs_with_highlight(nx_parent_graph, nx_iso_graph_2)



