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

    my_new_agent_2 = AssistantAgent(
        name="my_agent_2",
        model_client=llm,
        system_message="""Provide constructive feedback, so the text fits to the message in messanger (it is short and informative). 
        Respond with 'APPROVE' to when your feedbacks are addressed.
        """,
    )

    text_termination = TextMentionTermination("APPROVE")

    # Team of agents
    agent_new_team = RoundRobinGroupChat(
        [my_new_agent_1, my_new_agent_2], 
        termination_condition=text_termination
    )

    # ----------------------------------------
    # LOAD STATE of the agent
    team_state = joblib.load("team_state.pkl")
    print("-" * 30)
    print(team_state)  
    print("-" * 30)
    await agent_new_team.load_state(team_state)
    # ----------------------------------------

    stream = agent_new_team.run_stream(task="What was the last approved answer for my first question?")
    await Console(stream)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())