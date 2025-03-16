import os
from pprint import pprint
import anthropic
from dotenv import load_dotenv
load_dotenv()  

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# -----------------------------------------
# Initialize the chat history
messages = []
# -----------------------------------------

# First user query
messages.append({"role": "user", "content": "Tell me a joke."})

response1 = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    system="You are an AI assistant.",
    messages=messages
)

# Get assistant's response and add it to the history
messages.append({"role": "assistant", "content": response1.content[0].text})

print("--- Response text: ---")
print(response1.content[0].text)

# Second user query
messages.append({"role": "user", "content": "Repeat me please the previous joke."})

response2 = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    system="You are an AI assistant.",
    messages=messages
)

# Get assistant's response and add it to the history
messages.append({"role": "assistant", "content": response2.content[0].text})

print("--- Response text: ---")
print(response2.content[0].text)
