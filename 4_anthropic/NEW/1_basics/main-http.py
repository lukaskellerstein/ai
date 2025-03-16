import requests
import os
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY")

headers = {
    "Content-Type": "application/json", 
    "x-api-key": f"{api_key}",
    "anthropic-version": "2023-06-01"
}

prompt = "Tell me a joke."

# === Roles ===
# - user: The user message is used to provide input to the assistant.
# - assistant: The assistant message is used to provide output from the assistant.

# NO SYSTEM MESSAGE !!!
# But "system" parameter is available in the API

data = {
    "model": "claude-3-7-sonnet-20250219",
    "system": "You are an AI assistant.",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": prompt},
    ],
}

response = requests.post(
    "https://api.anthropic.com/v1/messages", headers=headers, json=data
)

print("--- Full response: ---")
pprint(response.json())
print("--- Response text: ---")
print(response.json()["content"][0]["text"])
