from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from dotenv import load_dotenv, find_dotenv
import asyncio
import os
from autogen_agentchat.conditions import TextMentionTermination
import joblib

load_dotenv(find_dotenv())  # read local .env file

# --------------------------------------------
# AI Agent
# 
# LOAD STATE
# --------------------------------------------

llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

async def main() -> None:
    my_new_agent_1 = AssistantAgent(
        name="my_agent_1",
        model_client=llm,
        system_message="You are a helpful AI assistant.",
    )

    # ----------------------------------------
    # LOAD STATE of the agent
    agent_state = joblib.load("agent_state.pkl")
    print("-" * 30)
    print(agent_state)  
    print("-" * 30)
    await my_new_agent_1.load_state(agent_state)
    # ----------------------------------------

    stream = my_new_agent_1.run_stream(task="What was the name of the car I asked about?")
    await Console(stream)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())