from pprint import pprint
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from dotenv import load_dotenv, find_dotenv
import asyncio
import os
from io import BytesIO
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image
import PIL
import requests

load_dotenv(find_dotenv())  # read local .env file


# --------------------------------------------
# Simple invoke of AI Agent
# 
# Mutli-modal message
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
    
    pil_image = PIL.Image.open("./image.jpg")
    img = Image(pil_image)

    # Message as object
    multi_modal_message = MultiModalMessage(content=["Can you describe the content of this image?", img], source="xxx")

    result = await agent.run(task=[multi_modal_message])
    # pprint(result)
    print(result.messages[len(result.messages) - 1].content)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())