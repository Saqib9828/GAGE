import json
import re
import csv
# List of special opcodes to analyze
special_opcodes = ["jmp", "call", "mov", "xor", "push", "pop", "add", "sub"]


def extract_opcode_frequency(code_block):
    opcode_frequency = {opcode: 0 for opcode in special_opcodes}
    opcode_frequency["ds:"] = 0
    opcode_frequency["hex_values"] = 0

    hex_pattern = re.compile(r'\b[0-9a-fA-F]+h\b')

    # Count opcode occurrences
    for instruction in code_block:
        for opcode in special_opcodes:
            if opcode in instruction:
                opcode_frequency[opcode] += 1

        # Count ds: occurrences
        if "ds:" in instruction:
            opcode_frequency["ds:"] += 1

        # Count hexadecimal value occurrences
        if hex_pattern.search(instruction):
            opcode_frequency["hex_values"] += 1
    return opcode_frequency

def normalize_frequency(opcode_frequency, total_instructions):
    normalized_frequency = {}
    for opcode, count in opcode_frequency.items():
        normalized_frequency[opcode] = count / total_instructions
    return normalized_frequency


def analyze_sample(sample_json):
    # Extract code blocks from JSON
    sample_code_blocks = sample_json['CPID_Features']

    # Initialize variables
    total_opcode_frequency = {opcode: 0 for opcode in special_opcodes}
    total_opcode_frequency["ds:"] = 0  # Add this line
    total_opcode_frequency["hex_values"] = 0  # Add this line
    total_instructions = 0

    # Analyze each code block
    for block_name, code_block in sample_code_blocks.items():
        opcode_frequency = extract_opcode_frequency(code_block)
        total_instructions += len(code_block)

        # Accumulate opcode frequencies
        for opcode, count in opcode_frequency.items():  # Modify this line
            total_opcode_frequency[opcode] += count  # Modify this line

    # Normalize opcode frequency
    normalized_frequency = normalize_frequency(total_opcode_frequency, total_instructions)

    return total_opcode_frequency, normalized_frequency

def save_to_csv(opcode_frequency, normalized_frequency, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Opcode/Operand", "Frequency", "Normalized Frequency"])
        # Write data
        for opcode, freq in opcode_frequency.items():
            norm_freq = normalized_frequency[opcode]
            writer.writerow([opcode, freq, norm_freq])

# Example usage:
if __name__ == '__main__':
    subgraph_file = "C:\\Users\\Saqib\\PycharmProjects\\GAGE\\expriment_out\\benign\\result_with_nodes_extracted_ex__1_benign_0a925bae354e3a051366c460f22821eeb15ea4730a4c904c24360da897bb3331__v1.tmp0.json"
    subgraph_file_quantitative = "C:\\Users\\Saqib\\PycharmProjects\\GAGE\\expriment_out\\benign\\result_with_nodes_extracted_ex__1_benign_0a925bae354e3a051366c460f22821eeb15ea4730a4c904c24360da897bb3331__v1.tmp0.csv"
    with open(subgraph_file, 'r') as file:
        data = json.load(file)
    opcode_frequency, normalized_frequency = analyze_sample(data)

    # Output the results
    print(f"Opcode Frequency: {opcode_frequency}")
    print(f"Normalized Frequency: {normalized_frequency}")

    save_to_csv(opcode_frequency, normalized_frequency, subgraph_file_quantitative)

