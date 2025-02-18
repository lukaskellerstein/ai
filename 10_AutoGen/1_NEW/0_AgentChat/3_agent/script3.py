from pprint import pprint
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from dotenv import load_dotenv, find_dotenv
import asyncio
import os

load_dotenv(find_dotenv())  # read local .env file

# --------------------------------------------
# Simple invoke of AI Agent
# 
# Multiple invocation
# YES - history of messages !!!!
# --------------------------------------------

llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

async def main() -> None:
    agent = AssistantAgent(
        name="my_assistant", 
        model_client=llm,
        system_message="You are a helpful AI assistant.")

    #--------------------
    # First invocation
    #--------------------
    result = await agent.run(task="Tell me a joke.")
    print(result.messages[len(result.messages) - 1].content)

    # stream = agent.run_stream(task="Tell me a joke.")
    # await Console(stream)

    #--------------------
    # Second invocation
    #--------------------
    result = await agent.run(task="What was the first joke you told me?")
    print(result.messages[len(result.messages) - 1].content)

    # stream = agent.run_stream(task="What was the first joke you told me?")
    # await Console(stream)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())