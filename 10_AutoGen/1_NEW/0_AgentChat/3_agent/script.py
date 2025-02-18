from pprint import pprint
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from dotenv import load_dotenv, find_dotenv
import asyncio
import os

load_dotenv(find_dotenv())  # read local .env file

# --------------------------------------------
# Simple invoke of AI Agent
# 
# Message as string
# Message as object
# --------------------------------------------

# AZURE ---------
# AZURE_API_KEY = os.getenv("AZURE_API_KEY")
# AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

# LLM
# llm = AzureOpenAIChatCompletionClient(
#             azure_deployment="gpt-4o-deployment-2",
#             azure_endpoint=AZURE_ENDPOINT,
#             api_key=AZURE_API_KEY,
#         )

# OPEN AI --------
llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

async def main() -> None:
    agent = AssistantAgent(
        name="my_assistant", 
        model_client=llm,
        system_message="You are a helpful AI assistant.")

    # Message as string
    result = await agent.run(task="Tell me a joke.")
    # pprint(result)
    print(result.messages[len(result.messages) - 1].content)

    # Message as object
    message = TextMessage(content="Tell me another joke.", source="xxx")
    result = await agent.run(task=message)
    # pprint(result)
    print(result.messages[len(result.messages) - 1].content)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())