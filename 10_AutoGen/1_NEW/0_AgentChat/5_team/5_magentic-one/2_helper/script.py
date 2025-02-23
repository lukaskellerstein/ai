import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_agentchat.ui import Console

async def example_usage():
    client = OpenAIChatCompletionClient(model="gpt-4o")
    m1 = MagenticOne(client=client)

    stream = m1.run_stream(task="Write a Python script to fetch data from an API.")
    result = await Console(stream)
    print(result)

if __name__ == "__main__":
    asyncio.run(example_usage())