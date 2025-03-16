from ollama import chat
from ollama import ChatResponse
from pprint import pprint

response1: ChatResponse = chat(
    model='llama3.2', 
    messages=[
        {
            'role': 'system',
            'content': 'You are an AI assistant.',
        },
        {
            'role': 'user',
            'content': 'Tell me a joke.',
        },
    ]
)

print("--- Response text: ---")
print(response1.message.content)

response2: ChatResponse = chat(
    model='llama3.2', 
    messages=[
        {
            'role': 'system',
            'content': 'You are an AI assistant.',
        },
        {
            'role': 'user',
            'content': "Repeat me please the previous joke.",
        },
    ]
)

print("--- Response text: ---")
print(response2.message.content)