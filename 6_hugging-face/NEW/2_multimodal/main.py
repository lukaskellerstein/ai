import os
from openai import OpenAI
from pprint import pprint
from encode import encode_image
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
	base_url="https://router.huggingface.co/hf-inference/v1",
	api_key=os.environ.get("HF_TOKEN"),
)

response = client.chat.completions.create(
	model="meta-llama/Llama-3.2-11B-Vision-Instruct", 
	messages=[
        {"role": "developer", "content": "You are an AI assistant."},
        {
            "role": "user",
            "content":  [
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image("./test_image.png")}",
                    }
                }
            ]
        },
    ],
	max_tokens=500,
)

print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
print(response.choices[0].message.content)