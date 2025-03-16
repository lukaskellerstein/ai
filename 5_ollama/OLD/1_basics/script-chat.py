import json
import requests


data = {
    "model": "mistral:v0.2",
    "messages": [
        {
        "role": "system",
        "content": "You are a helpful assistant with output as Shakespear."
        },
        {
        "role": "function",
        "content": "Hi"
        },
        {
        "role": "assistant",
        "content": "Hello Lukas."
        },
        {
        "role": "user",
        "content": "why is the sky blue?"
        }
    ],
    "stream": False
}

response = requests.post(
  "http://localhost:11434/api/chat", 
  data=json.dumps(data)
)
print(response.json())