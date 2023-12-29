import json

# functions 
def loadJSON(file):
    with open(file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

def saveJSON(file_path, data):
    with open(file_path, "w", encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)
        print(f"    |   JSON data has been saved to {file_path}")