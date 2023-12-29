import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
import os
import json
from tqdm import tqdm

def padding_blocks(blocks, expected_length=128, n_instructions=512):
    if not blocks:
        # If blocks is empty, create a dummy list of 1 list of 128 zeros
        dummy_block = [[0] * expected_length]
        blocks.append(dummy_block)

    for i in range(len(blocks)):
        if len(blocks[i]) < n_instructions:
            padding_needed = n_instructions - len(blocks[i])
            for _ in range(padding_needed):
                blocks[i].append([0] * expected_length)  # Pad with zero blocks
        elif len(blocks[i]) > n_instructions:
            # Reduce the elements to n_instructions by truncating
            blocks[i] = blocks[i][:n_instructions]

    # Process the blocks as before
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            if len(blocks[i][j]) < expected_length:
                # Pad the block with zeros to the expected length
                padding = expected_length - len(blocks[i][j])
                blocks[i][j].extend([0] * padding)  # Add zeros to the end
            elif len(blocks[i][j]) > expected_length:
                # If the block is longer than the expected length, truncate it
                blocks[i][j] = blocks[i][j][:expected_length]

    return blocks

def extract_encoded_features(blocks, autoencoder_model=load_model('models/AED_autoencoder_202310030112_Ori_v3.h5')):
    blocks_encoded_features = []
    for block in blocks:
        arr_block = np.array([block])
        encoded_features = autoencoder_model.layers[1](arr_block)
        blocks_encoded_features.append(encoded_features)
    return blocks_encoded_features

def blockFeatureAED(dir_path, output_dir):
    for root, _, files in os.walk(dir_path):
        print(root)
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                print("\t", file_path)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    nodes = data.get("Nodes", [])
                    edges = data.get("Edges", [])
                    new_nodes = []
                    for node in nodes:
                        new_node = {}
                        new_node["CPID"] = node.get("CPID", [])
                        new_node["Features"] = node.get("Features", [])
                        features_encoding = node.get("Features_encoding", [])
                        padded_blocks = padding_blocks([features_encoding])
                        collect_encoded_feature = extract_encoded_features(padded_blocks)
                        encoded_feature_list = collect_encoded_feature[0].numpy().tolist()  # Convert to list
                        new_node["AED_features"] = encoded_feature_list
                        new_nodes.append(new_node)

                     # Define the new file name (v3_oldfilename.json)
                    new_file_name = f"{file_name}"
                    new_file_path_str = output_dir + root.split(dir_path)[1] + "\\" + new_file_name
                    new_file_path = os.path.join(new_file_path_str)
                    print("new:", new_file_path)

                    # Create the directories if they do not exist
                    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

                    # Save the modified nodes back into the new JSON file
                    with open(new_file_path, 'w') as new_json_file:
                        json.dump({"Nodes": new_nodes, "Edges": edges}, new_json_file, indent=4)

    return "Features saved to JSON files."

directory_path = 'C:\\Users\Saqib\PycharmProjects\GAGE\\output\CEG_data'
new_output_dir = 'C:\\Users\Saqib\PycharmProjects\GAGE\\output_AED\\'
result = blockFeatureAED(directory_path, new_output_dir)
print(result)
