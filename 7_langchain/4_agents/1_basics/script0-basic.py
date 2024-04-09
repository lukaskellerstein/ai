from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
import pprint
import os
from dotenv import load_dotenv, find_dotenv
from helpers import printObject
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

_ = load_dotenv(find_dotenv())  # read local .env file

# Tools
tools = [TavilySearchResults(max_results=1)]



# Prompt ------------------------------------------------
prompt = hub.pull("hwchase17/openai-tools-agent")
printObject("prompt", prompt)


# Model ------------------------------------------------
# Choose the LLM that will drive the agent
# Only certain models support this
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
printObject("llm", llm)

# Agent ------------------------------------------------

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)
printObject("agent", agent)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what is LangChain?"})