from utility import *
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
import shutil
from tqdm import tqdm
from PalmTree.pre_trained_model.how2use import palmTreeEncoder

# constants
max_len_token = 6
opcodes_file_path = "data/opcodes.txt"
operands_file_path = "data/operands.txt"
with open(opcodes_file_path, "r") as file:
    unique_opcodes = [line.strip() for line in file]

with open(operands_file_path, "r") as file:
    unique_operands = [line.strip() for line in file]

# functions
def fileToCEG(file):
    # file 2 CEG
    data = loadJSON(file)

    nodes = []
    edges = []

    # Function to add an edge to the edges list
    def add_edge(source, destination, is_consequent, is_conditional, intra_edge):
        edge = {
            "Source": source,
            "Destination": destination,
            "is_consequent": is_consequent,
            "is_conditional": is_conditional,
            "intra_edge": intra_edge,
            "external_edge": not intra_edge
        }
        edges.append(edge)

    # Extract information from the JSON to construct nodes and edges
    functions = data["functions"]
    for function in functions:
        function_id = function["id"]
        blocks = function["blocks"]
        for block in blocks:
            block_id = block["id"]
            cpid = f"{function_id}_{block_id}"
            src = block["src"]

            # Create a node
            node = {
                "CPID": cpid,
                "Features": src
            }
            nodes.append(node)

            # Extract edges based on "call" within the block
            calls = block["call"]
            for call_id in calls:
                destination_cpid = f"{function_id}_{call_id}"

                # Define the edge type based on your criteria
                add_edge(cpid, destination_cpid, False, True, True)

            # If it's the last block in the function, add a consequent intra-edge
            if block_id == len(blocks) - 1:
                add_edge(cpid, f"{function_id}_{block_id + 1}", True, False, True)

        # Extract edges based on "call" within the function
        for call_id in function["call"]:
            destination_cpid = f"{call_id}_0"
            add_edge(f"{function_id}_{len(blocks) - 1}", destination_cpid, False, True, False)

    # Extract edges for consequent external edges between functions
    for i in range(len(functions) - 1):
        source_cpid = f"{functions[i]['id']}_{len(functions[i]['blocks']) - 1}"  # Last block of current function
        destination_cpid = f"{functions[i + 1]['id']}_0"  # First block of next function

        add_edge(source_cpid, destination_cpid, True, False, False)

    # Create the CEG JSON structure
    ceg_json = {
        "Nodes": nodes,
        "Edges": edges
    }

    # Convert the CEG data to JSON format and print it
    #ceg_json_str = json.dumps(ceg_json, indent=2)
    #print(ceg_json_str)
    return ceg_json

###################################################################
# encoding functions
# ------------------------------------------------------------------

def preprocess_features(features): # Function to preprocess the "Features" field
    preprocessed_features = []
    #print(features[0])
    for feature in features:
        feature.pop(0)
        preprocessed_feature = str(feature[0]) + " " +', '.join(feature[1:])
        preprocessed_features.append([preprocessed_feature])
    return preprocessed_features

def create_dictionary(ceg_data): # Initialize lists to store all opcodes and operands
    all_opcodes = []
    all_operands = []
    for node in ceg_data["Nodes"]:
        node["Features"] = preprocess_features(node["Features"])
    # Extract opcodes and operands from each instruction
    for node_entry in ceg_data["Nodes"]:
        for instruction_entry in node_entry["Features"]:
            instruction = instruction_entry[0].replace(",", "")
            parts = instruction.split(" ")
            if len(parts) >= 2:
                opcode = parts[0]
                all_opcodes.append(opcode)
                all_operands = all_operands + parts[1:]
    processed_operands = preprocess_operands(all_operands)
    all_opcodes = list(set(all_opcodes))
    all_processed_operands = list(set(processed_operands))
    return all_opcodes, all_processed_operands

def get_OpcodesOperands(ceg_data): # Initialize lists to store all opcodes and operands
    for node in ceg_data["Nodes"]:
        node["Features"] = preprocess_features(node["Features"])
    # Extract opcodes and operands from each instruction
    for node_entry in ceg_data["Nodes"]:
        process_inst = []
        for instruction_entry in node_entry["Features"]:
            instruction = instruction_entry[0].replace(",", "")
            parts = instruction.split(" ")
            opcode = parts[0]
            operands = []
            if len(parts) >= 1:
                operands = parts[1:]
            processed_operands = preprocess_operands(operands)
            process_inst.append([opcode, processed_operands])
        node_entry["processed_Features"] = process_inst
    return ceg_data

def encode_opcode(key):
    try:
        index = unique_opcodes.index(key)
        return index
    except ValueError:
        return -1  # Key not found in the list

def encode_operands(query):
    encoded_query = []
    word_to_index = {word: index for index, word in enumerate(unique_operands)}
    for word in query:
        if word in word_to_index:
            encoded_query.append(word_to_index[word])
    return encoded_query

def extract_unique_operands(word_list):
    unique_words = set()
    for word in word_list:
        words = word.split()
        unique_words.update(words)
    unique_word_list = list(unique_words)
    return unique_word_list

def is_integer_string(s):
    try:
        int(s)  # Try to convert the string to an integer
        return True
    except ValueError:
        return False

def preprocess_operands(operand_list):
    with open(opcodes_file_path, "r") as file: #take operands vocab
        relevant_operands = [line.strip() for line in file]

    processed_operands = []

    for operand in operand_list:
        operand = operand.translate(str.maketrans('', '', '[]()'))
        # Split the operand by spaces and commas to extract individual tokens
        tokens = re.split(r'[-,:_+ ]+', operand)

        # Initialize a list to store relevant tokens
        relevant_tokens = []

        # Iterate through tokens and check if they match relevant operands
        for token in tokens:
            # Check if the token is not empty
            if token:
                # Check if the token is a hexadecimal value (ends with 'h' or 'H')
                if token[-1].lower() == 'h' and len(token) > 1 and all(c in '0123456789ABCDEFabcdef' for c in token[:-1]):
                    relevant_tokens.append("hexvalue")  # Keep the hexadecimal value as-is
                elif token in relevant_operands:
                    relevant_tokens.append(token)
                elif is_integer_string(token):
                    relevant_tokens.append("const")
                elif len(token)>=max_len_token:
                    relevant_tokens.append("string_")
                else:
                    relevant_tokens.append("other_")

        # Join relevant tokens back into a single string
        processed_operand = ' '.join(relevant_tokens)

        # Append the processed operand to the result list
        processed_operands.append(processed_operand)

    return processed_operands

def print_operands_with_preprocessing(operand_list):
    print(f"{'Original Operand':<30} {'Preprocessed Operand':<30}")
    print("=" * 60)
    processed_operands = preprocess_operands(operand_list)
    for original, processed in zip(operand_list, processed_operands):
        print(f"{original:<30} {processed:<30}")


# ------------------------------------------------------------------
# end encoding functions
###################################################################

###################################################################
# Palm Tree encoder
# ------------------------------------------------------------------
def preprocess_features_palmTree(features):
    preprocessed_features = []
    for feature_list in features:
        feature_list.pop(0)
        preprocessed_feature = " ".join(feature_list)
        preprocessed_features.append(preprocessed_feature)
    return preprocessed_features

def feature_encoder_PalmTree(features):
    result = palmTreeEncoder(features).tolist()
    return result

def add_feature_encoder_PalmTre(ceg_json):
    for node in ceg_json["Nodes"]:
        node["Features"] = preprocess_features_palmTree(node["Features"])
    for node in ceg_json["Nodes"]:
        node["Features_encoding"] = feature_encoder_PalmTree(node["Features"])
    return ceg_json

def add_feature_encoder_ToFile_PalmTre(input_file, output_path):
    ceg = fileToCEG(input_file)
    ceg_json = add_feature_encoder_PalmTre(ceg)
    saveJSON(output_path, ceg_json)

# Function to check if a file size exceeds the specified limit (in bytes)
def is_file_size_exceeded(file_path, max_size_bytes):
    return os.path.getsize(file_path) > max_size_bytes


def process_json_file(file_path, max_size_bytes):
    # Check if the file size exceeds the limit
    if is_file_size_exceeded(file_path, max_size_bytes):
        print(f"Skipping {file_path} (File size exceeds {max_size_bytes} bytes)")
        return

    # Rest of your processing code
    folder_name = os.path.basename(os.path.dirname(file_path))
    output_dir = os.path.join("output/CEG_data", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    add_feature_encoder_ToFile_PalmTre(file_path, output_file)


def process_directory(directory, max_size_bytes):
    json_files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if
                  file.endswith(".json")]

    with tqdm(total=len(json_files), unit="file") as pbar:
        for file_path in json_files:
            process_json_file(file_path, max_size_bytes)
            pbar.update(1)

# ------------------------------------------------------------------
# end Palm Tree encoder
###################################################################

if __name__ == '__main__':
    # Specify the root directory to start searching for JSON files
    root_directory = "E:\\saqib_work1\\data\\data_GAGE"  
    max_file_size_bytes = 32 * 1024 * 1024  # 8MB
    process_directory(root_directory, max_file_size_bytes)
    #file = "data/origin_80__.tmp0.json"
    #output_path = "output/" + file
    #add_feature_encoder_ToFile_PalmTre(file, output_path)
    # Load the provided JSON data
    #ceg = fileToCEG(file)


    #ceg_json = loadJSON(output_path)
    #ceg_json = add_feature_encoder_PalmTre(ceg_json)
    print("done!")
    #secondceg_json = get_OpcodesOperands(ceg_json)
    #print(secondceg_json)
    #output_path = "output/secondceg.json"
    #saveJSON(output_path, secondceg_json)
    #print(encode_opcode("mov"))
    #encoded_query = encode_operands(["ebx", "eax", "cl"])
    #print(encoded_query)
    #print(all_operands)
    #print_operands_with_preprocessing(op)




