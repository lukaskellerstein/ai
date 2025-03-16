import requests
import os
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

headers = {
    "Content-Type": "application/json", 
    "Authorization": f"Bearer {api_key}"
}

data = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions", headers=headers, json=data
)

print("--- Full response: ---")
pprint(response.json())
print("--- Response text: ---")
print(response.json()["choices"][0]["message"]["content"])
