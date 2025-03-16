import os
from pprint import pprint
import anthropic
from dotenv import load_dotenv
load_dotenv()  

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

response1 = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    system="You are an AI assistant.",
    messages=[
        {"role": "user", "content": "Tell me a joke."},
    ]
)

print("--- Response text: ---")
print(response1.content[0].text)


response2 = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    system="You are an AI assistant.",
    messages=[
        {"role": "user", "content": "Repeat me please the previous joke."},
    ]
)

print("--- Response text: ---")
print(response2.content[0].text)
