import os
from pprint import pprint
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv() 

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response1 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "You are an AI assistant."},
        {
            "role": "user",
            "content": "Tell me a joke.",
        },
    ],
)

print("--- Response text: ---")
print(response1.choices[0].message.content)

response2 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "You are an AI assistant."},
        {
            "role": "user",
            "content": "Repeat me please the previous joke.",
        },
    ],
)

print("--- Response text: ---")
print(response2.choices[0].message.content)