from langchain import hub
from langchain.agents import AgentExecutor, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_functions import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.utils.function_calling import convert_to_openai_function
from helpers import printObject
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv, find_dotenv
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.callbacks import BaseCallbackHandler

_ = load_dotenv(find_dotenv())  # read local .env file



# Tools
tools = [TavilySearchResults(max_results=1)]



# prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
printObject("prompt", prompt)




# ----------------------
# Custom callback - to track what is happening during invoking
# ----------------------

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_start(self, _, prompts, *args, **kwargs) -> None:
        print(f"on_llm_start callback:")
        print("prompts")
        print(prompts)
        # print("args")
        # print(args)
        # print("kwargs")
        # print(kwargs)

    def on_chat_model_start(self, _, messages, *args, **kwargs) -> None:
        print(f"on_chat_model_start callback:")
        print("messages")
        print(messages)
        # print("args")
        # print(args)
        # print("kwargs")
        # print(kwargs)



# model
model = OllamaFunctions(model="mistral:v0.2", callbacks=[MyCustomHandler()])
printObject("llm", model)


# Agent ------------------------------------------------


converted_tools = []
for tool in tools:
    printObject(f"tool {tool}", tool)
    converted = convert_to_openai_function(tool)
    print(f"converted {tool}", converted)
    converted_tools.append(converted)

model_with_tools = model.bind(functions=converted_tools)
printObject("llm with tools", model_with_tools)

# agent = (
#         RunnablePassthrough.assign(
#             agent_scratchpad=lambda x: format_to_openai_function_messages(
#                 x["intermediate_steps"]
#             )
#         )
#         | prompt
#         | model_with_tools
#         | OpenAIFunctionsAgentOutputParser()
#     )

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | model_with_tools
    | OpenAIFunctionsAgentOutputParser()
)
printObject("agent", agent)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "whats the weather in New york?"})

print("--- RESULT ---")
print(result)