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

    my_agent_2 = AssistantAgent(
        name="my_agent_2",
        model_client=llm,
        system_message="""Provide constructive feedback, so the text fits to the message in messanger (it is short and informative). 
        Respond with 'APPROVE' to when your feedbacks are addressed.
        """,
    )

    text_termination = TextMentionTermination("APPROVE")

    # Team of agents
    agent_team = RoundRobinGroupChat(
        [my_agent_1, my_agent_2], 
        termination_condition=text_termination
    )

    stream = agent_team.run_stream(task="Is a Subaru Impreza STI a good car?")
    await Console(stream)

    # ----------------------------------------
    # SAVE STATE of the team
    team_state = await agent_team.save_state()
    print("-" * 30)
    print(team_state)  
    print("-" * 30)
    joblib.dump(team_state, "team_state.pkl")
    # ----------------------------------------

# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())