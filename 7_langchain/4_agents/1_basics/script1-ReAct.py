from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
import pprint
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI
from helpers import printObject

_ = load_dotenv(find_dotenv())  # read local .env file

# Tools
tools = [TavilySearchResults(max_results=1)]



# Prompt ------------------------------------------------
prompt = hub.pull("hwchase17/react")
printObject("prompt", prompt)


# Model ------------------------------------------------
# Choose the LLM to use
llm = OpenAI()
printObject("llm", llm)

# Agent ------------------------------------------------

# Construct the OpenAI Tools agent
agent = create_react_agent(llm, tools, prompt)
printObject("agent", agent)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what is LangChain?"})