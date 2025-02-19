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
# SAVE STATE
# --------------------------------------------

llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

async def main() -> None:
    my_agent_1 = AssistantAgent(
        name="my_agent_1",
        model_client=llm,
        system_message="You are a helpful AI assistant.",
    )

    stream = my_agent_1.run_stream(task="Is a Subaru Impreza STI a good car?")
    await Console(stream)

    # ----------------------------------------
    # SAVE STATE of the agent
    agent_state = await my_agent_1.save_state()
    print("-" * 30)
    print(agent_state)  
    print("-" * 30)
    joblib.dump(agent_state, "agent_state.pkl")
    # ----------------------------------------

# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())