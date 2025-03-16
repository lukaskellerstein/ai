import os
from pprint import pprint
import anthropic
from dotenv import load_dotenv
load_dotenv()  

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# === Roles ===
# - user: The user message is used to provide input to the assistant.
# - assistant: The assistant message is used to provide output from the assistant.

# NO SYSTEM MESSAGE !!!
# But "system" parameter is available in the API

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    system="You are an AI assistant.",
    messages=[
        {"role": "user", "content": "Tell me a joke."},
    ]
)

print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
print(response.content[0].text)
