import os
from pprint import pprint
import anthropic
from encode import encode_image
from dotenv import load_dotenv
load_dotenv() 

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encode_image("./test_image.png"),
                    },
                },
            ],
        }
    ],
)

print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
print(response.content[0].text)

