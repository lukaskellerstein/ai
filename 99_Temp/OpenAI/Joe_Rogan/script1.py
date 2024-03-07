# import json
# import re

# def convert_to_jsonl(file_path):
#     jsonl_data = []
#     with open(file_path, 'r') as file:
#         content = file.read()
#         matches = re.findall(r'\[(.*?)\] (.*?): (.*?)(?=\[\d|\Z)', content, re.DOTALL)
#         for match in matches:
#             time, speaker, text = match
#             jsonl_data.append(json.dumps({"time": time.strip(), "speaker": speaker.strip(), "text": text.strip()}))
    
#     with open('jre-1169-elon-musk-formatted-output.jsonl', 'w') as outfile:
#         for entry in jsonl_data:
#             outfile.write(entry + "\n")

# convert_to_jsonl('Podcast/jre-1169-elon-musk-formatted.txt')



# import re
# import json

# def convert_to_jsonl(file_path):
#     with open(file_path, 'r') as file:
#         data = file.read()

#     pattern = r"(\w+ \w+): \((\d{2}:\d{2})\) (.+?)(?=(?:\w+ \w+: \(\d{2}:\d{2}\)|$))"
#     matches = re.findall(pattern, data, re.DOTALL)

#     jsonl_data = []
#     for match in matches:
#         speaker, time, text = match
#         jsonl_data.append(json.dumps({"time": time.strip(), "speaker": speaker.strip(), "text": text.strip()}))

#     with open('jre-1470-elon-musk-formatted-output2.jsonl', 'w') as file:
#         file.write('\n'.join(jsonl_data))

# convert_to_jsonl('temp.txt')

import re
import json

def convert_to_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    pattern = r"(\w+ \w+): \((\d{2}:\d{2}:\d{2})\) (.+?)(?=(?:\w+ \w+: \(\d{2}:\d{2}:\d{2}\)|$))"
    matches = re.findall(pattern, data, re.DOTALL)

    jsonl_data = []
    for match in matches:
        speaker, time, text = match
        jsonl_data.append(json.dumps({"time": time.strip(), "speaker": speaker.strip(), "text": text.strip()}))

    with open('jre-1470-elon-musk-formatted-output2.jsonl', 'w') as file:
        file.write('\n'.join(jsonl_data))

convert_to_jsonl('temp.txt')