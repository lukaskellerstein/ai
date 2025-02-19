import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools import ag_tavily_tool

_llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

my_agent_researcher = AssistantAgent(
        name="my_researcher", 
        model_client=_llm, 
        system_message="You are a web researcher.",
        tools=[ag_tavily_tool]
    )