import json

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Path to the JSON file
train_file_paths = ['my_data/all_podcasts-train.json', "my_data/all_standups-train.json"]

# Loading the data
train_data = []

for file_path in train_file_paths:
    data = load_json(file_path)
    train_data.extend(data)
    print(f"Loaded {len(data)} records from {file_path}")



# Path to the JSON file
eval_file_paths = ['my_data/all_podcasts-eval.json', "my_data/all_standups-eval.json"]

# Loading the data
eval_data = []

for file_path in eval_file_paths:
    data = load_json(file_path)
    eval_data.extend(data)
    print(f"Loaded {len(data)} records from {file_path}")