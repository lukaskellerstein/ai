import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv() 

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.responses.create(
    model="gpt-4o",
    instructions="You are an AI assistant",
    input="Tell me a joke.",
)

print("--- Full response: ---")
print(response)
print("--- Response text: ---")
print(response.output_text)