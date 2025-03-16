import requests
import os
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

headers = {
    "Content-Type": "application/json", 
    "api-key": f"{os.environ.get("AZURE_OPENAI_API_KEY")}"
}

data = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
}

response = requests.post(
    "https://my-openai-4444.openai.azure.com/openai/deployments/gpt-4o-mini-deployment/chat/completions?api-version=2024-06-01", 
    headers=headers, 
    json=data
)

print("--- Full response: ---")
pprint(response.json())
print("--- Response text: ---")
print(response.json()["choices"][0]["message"]["content"])
