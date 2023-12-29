import json

query_file_name = "C:\\Users\\Saqib\\PycharmProjects\\GAGE\\expriment_out\\gamarue\\extracted_ex__1_gamarue_origin_2265__.tmp0.json"
feature_file_name =  "C:\\Users\\Saqib\\PycharmProjects\\GAGE\output_AED\\gamarue\\origin_2265__.tmp0.json"
output_file_subgraph = "C:\\Users\\Saqib\\PycharmProjects\\GAGE\\expriment_out\\gamarue\\result_with_nodes_extracted_ex__1_gamarue_origin_2265__.tmp0.json"

# Load data from query.json and feature.json
with open(query_file_name, "r") as query_file:
    query_data = json.load(query_file)

with open(feature_file_name, "r") as feature_file:
    feature_data = json.load(feature_file)

# Extract CPID values from query.json
cpid_values = set()
for record in query_data:
    cpid_values.add(record["Source"])
    cpid_values.add(record["Destination"])

# Create a dictionary to store CPID and their corresponding features
cpid_features = {}
for node in feature_data["Nodes"]:
    cpid = node["CPID"]
    if cpid in cpid_values:
        features = node["Features"]
        cpid_features[cpid] = features

# Create a new JSON with CPID and their Features
new_json_data = {"CPID_Features": cpid_features}

# Save the new JSON file
with open(output_file_subgraph, "w") as result_file:
    json.dump(new_json_data, result_file, indent=4)

print("CPID features saved to: ", output_file_subgraph)
