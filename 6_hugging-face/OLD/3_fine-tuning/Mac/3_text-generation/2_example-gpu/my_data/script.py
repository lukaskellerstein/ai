import json

# Load the JSON file
with open('all_standups-eval.json', 'r') as f:
    data = json.load(f)

# Initialize the expected role
expected_role = "user"

problems = 0
# Iterate over the conversations
for i, conversation in enumerate(data):
    # Check if the role matches the expected role
    if conversation["role"] != expected_role:
        if problems < 10:
            print(f"Error at index {i}: {conversation['content']}")
        problems += 1
    # Swap the expected role
    expected_role = "assistant" if expected_role == "user" else "user"