import os
from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
	base_url="https://router.huggingface.co/hf-inference/v1",
	api_key=os.environ.get("HF_TOKEN"),
)

response = client.chat.completions.create(
	model="meta-llama/Llama-3.2-3B-Instruct", 
	messages=[
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
	max_tokens=500,
)

print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
print(response.choices[0].message.content)