import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools import ag_tavily_tool

_llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

my_agent_news_analyst = AssistantAgent(
        name="my_news_analyst", 
        model_client=_llm, 
        handoffs=["my_planner"],
        tools=[ag_tavily_tool],
        system_message="""You are a news analyst.
Gather and analyze relevant news using the web tool.
Summarize key market insights from news.
Always handoff back to planner when analysis is complete.""",
    )