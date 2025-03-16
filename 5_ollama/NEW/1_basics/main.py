from ollama import chat
from ollama import ChatResponse
from pprint import pprint

response: ChatResponse = chat(
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

print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
print(response.message.content)