from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
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
# Team loop finishes when: MAX TURNS
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
        system_message="You are a helpful AI assistant.",
    )

    my_agent_2 = AssistantAgent(
        name="my_agent_2",
        model_client=llm,
        system_message="Provide constructive feedback.",
    )

    user_proxy = UserProxyAgent("user_proxy", input_func=input)

    # Team of agents
    agent_team = RoundRobinGroupChat(
        [my_agent_1, my_agent_2, user_proxy], 
        max_turns=10 
    )

    # LOOP -----
    # Will end only when MAX_TURNS is reached. No matter what user writes.
    # stream = agent_team.run_stream(task="Is a Subaru Impreza STI a good car?")
    # await Console(stream)

    # CONVERSATION -----
    # Will end when user writes 'exit', BUT AFTER MAX_TURNS is reached.
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