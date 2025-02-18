from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

_ = load_dotenv(find_dotenv())  # read local .env file

# Prompt ------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. 
            Your goal is to either create a plan for a given objective or to adjust existing plan based on latest informations (past_steps) from the team.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Model ------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Agent ------------------------------------------------
agent = (
    prompt
    | llm
    | OpenAIToolsAgentOutputParser()
)

# Agent executor
planner_agent = AgentExecutor(agent=agent, verbose=False)
# ----------------------------------