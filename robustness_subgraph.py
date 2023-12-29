import json

query_file_name = "C:\\Users\Saqib\PycharmProjects\GAGE\\robustness_files\\bladabindi\\gage_cfg\\ex_bladabindi_origin_3580__.tmp0.json"
feature_file_name =  "C:\\Users\\Saqib\\PycharmProjects\\GAGE\output_AED\\bladabindi\\origin_3580__.tmp0.json"
output_file_subgraph = "C:\\Users\\Saqib\\PycharmProjects\\GAGE\\robustness_files\\bladabindi\\gage_cfg\\result_for_robustness_ex_bladabindi_origin_3580__.tmp0.json"

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
AED_features = {}
for node in feature_data["Nodes"]:
    cpid = node["CPID"]
    if cpid in cpid_values:
        AED_features[cpid] = node["AED_features"]

# Create a new JSON with CPID and their Features
new_json_data = {"AED_features": AED_features}

# Save the new JSON file
with open(output_file_subgraph, "w") as result_file:
    json.dump(new_json_data, result_file, indent=4)

print("AED_features saved to: ", output_file_subgraph)
