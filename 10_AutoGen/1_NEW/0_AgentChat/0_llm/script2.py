from pprint import pprint
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from autogen_core.models import UserMessage
from dotenv import load_dotenv, find_dotenv
import asyncio
import os

load_dotenv(find_dotenv())  # read local .env file

# --------------------------------------------
# Simple invoke of LLM
# 
# Multiple invocation
# NO history of messages !!!!
# --------------------------------------------

llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Main function
async def main() -> None:

    #--------------------
    # First invocation
    #--------------------
    messages = [
        UserMessage(content="Tell me a joke.", source="xxx"),
    ]
    
    result = await llm.create(messages=messages)
    # pprint(result)
    print(result.content)

    #--------------------
    # Second invocation
    #--------------------
    messages = [
        UserMessage(content="What was the first joke you told me?", source="xxx"),
    ]

    result = await llm.create(messages=messages)
    # pprint(result)
    print(result.content)



# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())