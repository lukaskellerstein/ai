import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools import ag_tavily_tool

_llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

my_agent_writer = AssistantAgent(
        name="my_writer", 
        model_client=_llm, 
        handoffs=["my_planner"],
        tools=[ag_tavily_tool],
        system_message="""You are a financial report writer.
Compile research findings into clear, concise reports.
Always handoff back to planner when writing is complete.""",
)