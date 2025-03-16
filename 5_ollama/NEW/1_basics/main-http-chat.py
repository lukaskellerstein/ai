import json
import requests
from pprint import pprint

data = {
    "model": "llama3.2",
    "messages": [
        {
            "role": "system",
            "content": 'You are an AI assistant.'
        },
        {
            'role': 'user',
            'content': 'Tell me a joke.',
        },
    ],
    "stream": False
}

response = requests.post(
  "http://localhost:11434/api/chat", 
  data=json.dumps(data)
)

print("--- Full response: ---")
pprint(response.json())
print("--- Response text: ---")
print(response.json()['message'])