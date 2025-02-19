import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools import ag_repl_tool

_llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

my_agent_coder = AssistantAgent(
        name="my_coder", 
        model_client=_llm, 
        system_message="You are a coder, you can write any code, prefered language is Python.",
        tools=[ag_repl_tool]
    )