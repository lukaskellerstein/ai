from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from dotenv import load_dotenv, find_dotenv
import asyncio
import os
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.base import Handoff

load_dotenv(find_dotenv())  # read local .env file

llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

async def main() -> None:
    # Define an agent
    my_agent_1 = AssistantAgent(
        name="my_agent_1",
        model_client=llm,
        handoffs=[Handoff(target="user", message="Transfer to user.")],
        system_message="If you cannot complete the task, transfer to user. Otherwise, when finished, respond with 'TERMINATE'.",
    )

    # user_proxy = UserProxyAgent("user_proxy", input_func=input)

    handoff_termination = HandoffTermination(target="user")
    text_termination = TextMentionTermination("TEMINATE")

    # Define a team with maximum auto-gen turns of 10.
    agent_team = RoundRobinGroupChat(
        [my_agent_1], 
        termination_condition=text_termination | handoff_termination
    )

    # Question 1 (stream) ---
    stream1 = agent_team.run_stream(task="What is the weather in Prague right now?")
    await Console(stream1)

    stream2 = agent_team.run_stream(task="The weather in Prague is cold (-10 degrees) but sunny.")
    await Console(stream2)

# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())