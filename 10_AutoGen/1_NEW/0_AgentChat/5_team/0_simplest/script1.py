from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from dotenv import load_dotenv, find_dotenv
import asyncio
import os
from autogen_agentchat.conditions import TextMentionTermination

load_dotenv(find_dotenv())  # read local .env file

# --------------------------------------------
# Team of AI Agents
# 
# Finishes when: MAX TURNS
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

    my_agent_2 = AssistantAgent(
        name="my_agent_2",
        model_client=llm,
        system_message="Provide constructive feedback.",
    )

    # Team of agents
    agent_team = RoundRobinGroupChat(
        [my_agent_1, my_agent_2], 
        max_turns=10 
    )

    stream = agent_team.run_stream(task="Is a Subaru Impreza STI a good car?")
    await Console(stream)

# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())