import requests
import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd

_ = load_dotenv(find_dotenv())

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model_id = "sentence-transformers/all-MiniLM-L6-v2"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}


def query(texts):
    response = requests.post(
        api_url,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}},
    )
    return response.json()


text = """	
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

response = query(text)
print(response)

# pandas dataframe
embeddingsDf = pd.DataFrame(response)
print(embeddingsDf)
