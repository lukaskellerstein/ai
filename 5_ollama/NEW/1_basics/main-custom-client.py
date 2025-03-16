from ollama import Client
from pprint import pprint
client = Client(host='http://localhost:11434')

response = client.chat(
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