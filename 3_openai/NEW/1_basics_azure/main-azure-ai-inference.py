import os
from dotenv import load_dotenv
from pprint import pprint
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
load_dotenv()

client = ChatCompletionsClient(
    endpoint="https://my-openai-4444.openai.azure.com/openai/deployments/gpt-4o-mini-deployment",
    credential=AzureKeyCredential(os.environ.get("AZURE_OPENAI_API_KEY")),
)

response = client.complete(
    messages=[
        SystemMessage(content="You are an AI assistant."),
        UserMessage(content="Tell me a joke."),
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model="gpt-4o-mini"
)
print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
print(response.choices[0].message.content)