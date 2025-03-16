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

prompt = "Tell me a joke."

data = {
    "model": "gpt-4o",
    "instructions": "You are an AI assistant.",
    "input": "Tell me a joke.",
}

response = requests.post(
    "https://api.openai.com/v1/responses", headers=headers, json=data
)

print("--- Full response: ---")
pprint(response.json())
print("--- Response text: ---")
print(response.json()["output"][0]["content"][0]["text"])
