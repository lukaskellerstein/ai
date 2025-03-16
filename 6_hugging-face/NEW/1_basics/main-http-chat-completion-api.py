import requests
import os
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

headers = {
    "Content-Type": "application/json", 
    "Authorization": f"Bearer {os.environ.get("HF_TOKEN")}"
}

data = {
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
    "max_tokens": 500,
    "steam": False,
}

response = requests.post(
    "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.2-3B-Instruct/v1/chat/completions", 
    headers=headers, 
    json=data
)

print("--- Full response: ---")
pprint(response.json())
print("--- Response text: ---")
print(response.json()["choices"][0]["message"]["content"])
