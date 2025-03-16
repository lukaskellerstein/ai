from ollama import chat
from ollama import ChatResponse
from pprint import pprint
from encode import encode_image

response: ChatResponse = chat(
    model='llama3.2-vision', 
    messages=[
        {"role": "system", "content": "You are an AI assistant."},
        {
            "role": "user",
            "content":  "What is in this image?",
            "images": [encode_image("./test_image.png")],
        },
    ],
)

print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
print(response.message.content)