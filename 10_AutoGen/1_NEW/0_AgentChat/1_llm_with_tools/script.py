from pprint import pprint
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from autogen_core.models import UserMessage
from autogen_core.tools import FunctionTool
from dotenv import load_dotenv, find_dotenv
import asyncio
import os

load_dotenv(find_dotenv())  # read local .env file

# --------------------------------------------
# Simple invoke of LLM
# 
# Function calling
# Tool usage
# --------------------------------------------

llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define a tool
async def get_weather(city: str) -> str:
    return f"The weather in {city} is 73 degrees and Sunny."

weather_tool = FunctionTool(get_weather, description="Get the weather.")

async def get_stock_price(ticker: str) -> str:
    return f"The stock {ticker} price is 409 USD per share."

stock_price_tool = FunctionTool(get_stock_price, description="Get the stock price.")


# Main function
async def main() -> None:
    messages = [
        UserMessage(content="What is weather in Prague?", source="xxx"),
    ]
    
    result = await llm.create(
        messages=messages,
        tools=[weather_tool, stock_price_tool]
    )
    # pprint(result)
    print(result.content)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())