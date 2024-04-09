from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
import pprint
import os
from dotenv import load_dotenv, find_dotenv
from helpers import printObject
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

_ = load_dotenv(find_dotenv())  # read local .env file

# Tools
tools = [TavilySearchResults(max_results=1)]



# Prompt ------------------------------------------------
# Adapted from https://smith.langchain.com/hub/hwchase17/openai-tools-agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. You may not need to use tools for every query - the user may just want to chat!",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
printObject("prompt", prompt)


# Model ------------------------------------------------
# Choose the LLM that will drive the agent
# Only certain models support this
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
printObject("llm", llm)

# Agent ------------------------------------------------

converted_tools = []
for tool in tools:
    printObject(f"tool {tool}", tool)
    converted = convert_to_openai_tool(tool)
    print(f"converted {tool}", converted)
    converted_tools.append(converted)

llm_with_tools = llm.bind(tools=converted_tools)
printObject("llm with tools", llm_with_tools)

agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            )
        )
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
printObject("agent", agent)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what is LangChain?"})