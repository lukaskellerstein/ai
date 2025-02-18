from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
import pprint
import os
from dotenv import load_dotenv, find_dotenv
from helpers import printObject

_ = load_dotenv(find_dotenv())  # read local .env file

# Tools
tools = [TavilySearchResults(max_results=1, name="Intermediate Answer")]



# Prompt ------------------------------------------------
prompt = hub.pull("hwchase17/self-ask-with-search")
printObject("prompt", prompt)


# Model ------------------------------------------------
# Choose the LLM that will drive the agent
# Only certain models support this
llm = OpenAI()
printObject("llm", llm)

# Agent ------------------------------------------------

# Construct the OpenAI Tools agent
agent = create_self_ask_with_search_agent(llm, tools, prompt)
printObject("agent", agent)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

agent_executor.invoke({"input": "What is the hometown of the reigning men's U.S. Open champion?"})