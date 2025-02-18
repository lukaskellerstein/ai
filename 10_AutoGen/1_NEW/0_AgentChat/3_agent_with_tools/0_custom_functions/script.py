from pprint import pprint
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from dotenv import load_dotenv, find_dotenv
from autogen_agentchat.ui import Console
import asyncio
import os

load_dotenv(find_dotenv())  # read local .env file

# --------------------------------------------
# AI Agent
# 
# Tool as function
#
# LLM can select the tool
# Agent runs the tool
# LLM can process the result from tool and provide the answer
# --------------------------------------------

llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define a tool
async def get_weather(city: str) -> str:
    return f"The weather in {city} is 73 degrees and Sunny."

async def get_stock_price(ticker: str) -> str:
    return f"The stock {ticker} price is 409 USD per share."

# Main function
async def main() -> None:
    agent = AssistantAgent(
        name="my_assistant", 
        model_client=llm, 
        system_message="You are a helpful AI assistant.",
        tools=[get_weather, get_stock_price]
    )

    stream = agent.run_stream(task="What is stock price for AAPL?")
    await Console(stream)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())