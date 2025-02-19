import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

_llm = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

my_agent_planner = AssistantAgent(
    name="my_planner",
    # description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=_llm,
    handoffs=["my_coder", "my_financial_analyst", "my_news_analyst", "my_writer"],
    system_message="""You are a research planning coordinator.
Coordinate market research by delegating to specialized agents:
- Financial Analyst: For stock data analysis
- News Analyst: For news gathering and analysis
- Writer: For compiling final report
- Coder: For software development tasks. Saving charts into files ... etc.
Always send your plan first, then handoff to appropriate agent.
Always handoff to a single agent at a time.
Use TERMINATE when research is complete.""",
)