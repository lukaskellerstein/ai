from langchain import hub
from langchain.agents import AgentExecutor, tool
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain_community.chat_models import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from helpers import printObject

_ = load_dotenv(find_dotenv())  # read local .env file


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
printObject("prompt", prompt)



# model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# model = Ollama(model="mistral:v0.2")

# -------------------------------
# Tools
# -------------------------------

tools = [TavilySearchResults(max_results=1)]

# @tool
# def search(query: str) -> str:
#     """Search things about current events."""
#     return "32 degrees"

# @tool
# def get_word_length(word: str) -> int:
#     """Returns the length of a word."""
#     return len(word)


# tools = [search, get_word_length]

# # Logic for going from intermediate steps to a string to pass into model
# # This is pretty tied to the prompt
# def convert_intermediate_steps(intermediate_steps):
#     log = ""
#     for action, observation in intermediate_steps:
#         log += (
#             f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
#             f"</tool_input><observation>{observation}</observation>"
#         )
#     return log


# # Logic for converting tools to string to go in prompt
# def convert_tools(tools):
#     return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])


converted_tools = []
for tool in tools:
    printObject(f"tool {tool}", tool)
    converted = convert_to_openai_tool(tool)
    print(f"converted {tool}", converted)
    converted_tools.append(converted)

llm_with_tools = llm.bind(tools=converted_tools)


# -------------------------------
# Agent
# -------------------------------
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
printObject("agent", agent)

# Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# -------------------------------
# Run
# -------------------------------
agent_executor.invoke({"input": "whats the weather in New york?"})