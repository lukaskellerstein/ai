import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
)

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools import ShellTool, HumanInputRun
from langchain_experimental.utilities import PythonREPL

_ = load_dotenv(find_dotenv())  # read local .env file

# YOUR_POD_ID = "gbxz2a216watit"
# YOUT_POD_PORT = 8080
# API_URL = f"https://{YOUR_POD_ID}-{YOUT_POD_PORT}.proxy.runpod.net/v1"


# ----------------------
# Helper functions
# ----------------------
def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)


python_repl = PythonREPL()

# ----------------------
# Tools
# ----------------------
# fs_tools = FileManagementToolkit(
#     root_dir="/home/lukas/Projects/Temp",
#     selected_tools=["read_file", "list_directory"],
# ).get_tools()

tools = [
    TavilySearchResults(max_results=1),
    # ShellTool(),
    # *fs_tools,
    # HumanInputRun(input_func=get_input),
]

# ----------------------
# MODEL
# = Mistral v0.3,  Mistral v0.3 Instruct
# NO HISTORY !!!
# ----------------------

# Chat is possible to use with function calls
llm = ChatOpenAI(
    model="gpt-4-turbo",
    # base_url=API_URL,
)
# add tools to the model
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")
# tools_formatted = [convert_to_openai_tool(t) for t in tools]
# ----------------------

# Invoke 1
print("---- Invoke 1 ----")
prompt = [
    {
        "role": "user",
        "content": "What would be a good company name for a company that makes colorful socks?",
    },
]
result = llm_with_tools.invoke(prompt)
# result = llm.invoke(prompt, tools=tools_formatted)
print("Result:")
print(type(result))
print(result)

# Invoke 2 = NO HISTORY
print("---- Invoke 2 ----")
prompt = [
    {
        "role": "user",
        "content": "What is AutoGen?",
    },
]
result = llm_with_tools.invoke(prompt)
# result = llm.invoke(prompt, tools=tools_formatted)
print("Result:")
print(type(result))
print(result)

# Invoke 3 = NO HISTORY
print("---- Invoke 3 ----")
prompt = [
    {
        "role": "user",
        "content": "What date is today?",
    },
]
result = llm_with_tools.invoke(prompt)
# result = llm.invoke(prompt, tools=tools_formatted)
print("Result:")
print(type(result))
print(result)

# Invoke 4 = NO HISTORY
print("---- Invoke 4 ----")
prompt = [
    {
        "role": "user",
        "content": "Search for the best pizza in Prague.",
    },
]
result = llm_with_tools.invoke(prompt)
# result = llm.invoke(prompt, tools=tools_formatted)
print("Result:")
print(type(result))
print(result)
