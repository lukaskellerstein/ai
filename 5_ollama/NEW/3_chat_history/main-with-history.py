from ollama import chat
from ollama import ChatResponse
from pprint import pprint


# -----------------------------------------
# Initialize the chat history
messages = [
    {"role": "system", "content": "You are an AI assistant."}  # 'system' role is the correct one
]
# -----------------------------------------

# First user query
messages.append({"role": "user", "content": "Tell me a joke."})

response1: ChatResponse = chat(
    model='llama3.2', 
    messages=messages
)

# Get assistant's response and add it to the history
messages.append({"role": "assistant", "content": response1.message.content})

print("--- Response text: ---")
print(response1.message.content)

messages.append({"role": "user", "content": "Repeat me please the previous joke."})

response2: ChatResponse = chat(
    model='llama3.2', 
    messages=messages
)

# Get assistant's response and add it to the history
messages.append({"role": "assistant", "content": response2.message.content})

print("--- Response text: ---")
print(response2.message.content)