import os
from openai import AzureOpenAI
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

client = AzureOpenAI(
    api_version="2023-07-01-preview",
    azure_endpoint="https://my-openai-4444.openai.azure.com/",
    azure_deployment="gpt-4o-mini-deployment",
    azure_ad_token=os.environ.get("AZURE_OPENAI_API_KEY")
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
)

print("--- Full response: ---")
pprint(response.to_json())
print("--- Response text: ---")
print(response.choices[0].message.content)