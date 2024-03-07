import json
import itertools

def transform_jsonl(file_path):
    system_message = {"role": "system", "content": "This AI simulates Joe Rogan, the podcaster."}
    transformed_data = []
    with open(file_path, 'r') as file:
        while True:
            # Get the next two lines
            lines = list(itertools.islice(file, 2))

            # If the lines list is empty, we've reached the end of the file
            if not lines:
                break

            # Process the two lines
            for i, line in enumerate(lines):
                data = json.loads(line)
                if data["speaker"] == "Joe Rogan":
                    role = "assistant"
                else:
                    role = "user"
                if i == 0:
                    role1, text1 = role, data["text"]
                else:
                    role2, text2 = role, data["text"]

            if role1 != role2:
                transformed_data.append({"messages": [system_message, {"role": role1, "content": text1}, {"role": role2, "content": text2}]})
    
    with open('jre-1470-elon-musk-formatted-transformed_output.jsonl', 'w') as outfile:
        for data in transformed_data:
            outfile.write(json.dumps(data) + '\n')

transform_jsonl('jre-1470-elon-musk-formatted-output.jsonl')