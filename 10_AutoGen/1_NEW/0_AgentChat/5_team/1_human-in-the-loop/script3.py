from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from dotenv import load_dotenv, find_dotenv
import asyncio
import os
from autogen_agentchat.conditions import HandoffTermination
from autogen_agentchat.base import Handoff

load_dotenv(find_dotenv())  # read local .env file

# --------------------------------------------
# Team of AI Agents
# 
# Team loop finishes when: USER AS TOOL
# 
# Conversation finishes when: USER says 'exit'
# --------------------------------------------
llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

async def main() -> None:
    my_agent_1 = AssistantAgent(
        name="my_agent_1",
        model_client=llm,
        handoffs=[Handoff(target="user", message="Transfer to user.")],
        system_message="You are a helpful AI assistant.",
    )

    my_agent_2 = AssistantAgent(
        name="my_agent_2",
        model_client=llm,
        handoffs=[Handoff(target="user", message="Transfer to user.")],
        system_message="Provide constructive feedback.",
    )

    # Team of agents
    agent_team = RoundRobinGroupChat(
        [my_agent_1, my_agent_2], 
        termination_condition=HandoffTermination(target="user") 
    )

    # LOOP -----
    # # fist loop
    # stream = agent_team.run_stream(task="Is a Subaru Impreza STI a good car?")
    # await Console(stream)

    # # second loop
    # stream = agent_team.run_stream(task="Is it faster than Mitsubishi Lancer Evolution?")
    # await Console(stream)

    # CONVERSATION -----
    # Will end when user writes 'exit', BUT AFTER "USER AS TOOL" is needed.
    while True:
        # Get user input from the console.
        user_input = input("Enter a message (type 'exit' to leave): ")
        if user_input.strip().lower() == "exit":
            break
        # Run the team and stream messages to the console.
        stream = agent_team.run_stream(task=user_input)
        await Console(stream)

# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())