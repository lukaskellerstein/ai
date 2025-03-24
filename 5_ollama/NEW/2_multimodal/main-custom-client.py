from ollama import Client
from pprint import pprint
from encode import encode_image

client = Client(host='http://localhost:11434')

response = client.chat(
    model='llama3.2-vision', 
    messages=[
        {
            'role': 'system',
            'content': 'You are an AI assistant.',
        },
        {
            "role": "user",
            "content":  "What is in this image?",
            "images": [encode_image("./test_image.png")],
        },
    ]
)

print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
print(response.message.content)