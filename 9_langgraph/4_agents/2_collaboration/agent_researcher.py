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

# Tools
tools = [TavilySearchResults(max_results=3)]


# Prompt ------------------------------------------------
# Adapted from https://smith.langchain.com/hub/hwchase17/openai-tools-agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a web researcher. If you or any of the other assistants have the final answer or deliverable, prefix your response with [[[[FINAL ANSWER]]]] so the team knows to stop.",
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Model ------------------------------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(tools=tools, tool_choice="auto")

# Agent ------------------------------------------------
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

# Agent executor
researcher_agent = AgentExecutor(agent=agent, tools=tools, verbose=True)
