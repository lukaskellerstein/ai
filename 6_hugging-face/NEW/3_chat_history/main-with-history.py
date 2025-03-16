import os
from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
	base_url="https://router.huggingface.co/hf-inference/v1",
	api_key=os.environ.get("HF_TOKEN"),
)

# -----------------------------------------
# Initialize the chat history
messages = [{"role": "system", "content": "You are an AI assistant."}]
# -----------------------------------------

# First user query
messages.append({"role": "user", "content": "Tell me a joke."})

response1 = client.chat.completions.create(
	model="meta-llama/Llama-3.2-3B-Instruct", 
	messages=messages,
	max_tokens=500,
)

# Get assistant's response and add it to the history
messages.append({"role": "assistant", "content": response1.choices[0].message.content})

print("--- Response text: ---")
print(response1.choices[0].message.content)

# Second user query
messages.append({"role": "user", "content": "Repeat me please the previous joke."})

response2 = client.chat.completions.create(
	model="meta-llama/Llama-3.2-3B-Instruct", 
	messages=messages,
	max_tokens=500,
)

# Get assistant's response and add it to the history
messages.append({"role": "assistant", "content": response2.choices[0].message.content})

print("--- Response text: ---")
print(response2.choices[0].message.content)