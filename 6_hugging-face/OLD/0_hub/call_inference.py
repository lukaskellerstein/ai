import requests
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())

hf_token = os.getenv("HF_TOKEN")

API_URL = (
    "https://api-inference.huggingface.co/models/lukaskellerstein/mistral-7b-lex-16bit"
)
headers = {"Authorization": f"Bearer {hf_token}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query(
    {
        "inputs": "Can you please let us know more details about your ",
    }
)
print(output)
