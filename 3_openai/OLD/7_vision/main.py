from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see on the picture?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://mystorage456789.blob.core.windows.net/img/Screenshot_4.png"
                    },
                },
            ],
        },
    ],
)

print(response)
