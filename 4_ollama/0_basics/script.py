import json
import requests


data = {
    "model": "mistral:v0.2",
    "prompt": "Why is the sky blue?",
    "stream": False
}

response = requests.post(
  "http://localhost:11434/api/generate", 
  data=json.dumps(data)
)
print(response.json())