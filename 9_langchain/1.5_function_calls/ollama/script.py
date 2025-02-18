from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.utils.function_calling import convert_to_openai_function
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

# ----------------------
# Tools
# ----------------------
tools = [TavilySearchResults(max_results=1)]


# ----------------------
# Prompt
# ----------------------

# prompt 1 - string
# prompt = "What would be a good company name for a company that makes colorful socks?"

# prompt 2 - array
prompt = [
    {
        "role": "system",
        "content": "You are a helpful assistant with output as Shakespear.",
    },
    {
        "role": "user",
        "content": "What would be a good company name for a company that makes colorful socks?",
    },
]

# prompt 3 - array
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "You are AI assistant"),
#     ("human", "hi"),
#     ("ai", "hello {name}"),
#     ("human", "What is a good name for a company that makes {product}?"),
# ])

# prompt = prompt_template.format_messages(
#     name="Lukas",
#     product="colorful socks"
# )

# prompt 4 - array
# prompt = [
#     ("system", "You are AI assistant"),
#     ("human", "hi"),
#     ("ai", "hello Lukas"),
#     ("human", "What is a good name for a company that makes socks?"),
# ]


print("---- Prompt ----")
print(type(prompt))
print(prompt)



# ----------------------
# Custom callback - to track what is happening during invoking
# ----------------------


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_start(self, _, prompts, *args, **kwargs) -> None:
        print(f"Callback on_llm_start:")
        print("prompts")
        print(prompts)
        # print("args")
        # print(args)
        # print("kwargs")
        # print(kwargs)

    def on_chat_model_start(self, _, messages, *args, **kwargs) -> None:
        print(f"Callback on_chat_model_start:")
        print("messages")
        print(messages)
        # print("args")
        # print(args)
        # print("kwargs")
        # print(kwargs)



# ----------------------
# MODEL
# = Mistral v0.2,  Mistral Instruct
# NO HISTORY !!!
# ----------------------

# LLM is not possible to use with function calls
# llm = Ollama(model="mistral:v0.2", callbacks=[MyCustomHandler()]) # returns TEXT
# llm = Ollama(model="mistral:instruct", callbacks=[MyCustomHandler()]) # returns TEXT
# add tools to the model
# llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

# Chat is NOT possible to use with function calls
# llm = ChatOllama(model="mistral:v0.2", callbacks=[MyCustomHandler()]) # returns MESSAGE object
# llm = ChatOllama(model="mistral:instruct", callbacks=[MyCustomHandler()]) # returns MESSAGE object
# add tools to the model
# llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

# Special class needed = OllamaFunctions
llm = OllamaFunctions(model="mistral:v0.2", callbacks=[MyCustomHandler()]) # returns TEXT
# add tools to the model
llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

# ----------------------

# Invoke 1
print("---- Invoke 1 ----")
result = llm_with_tools.invoke(prompt)
print("Result:")
print(type(result))
print(result)

# Invoke 2 = NO HISTORY
print("---- Invoke 2 ----")
prompt = "Give me another 5."
result = llm_with_tools.invoke(prompt)
print("Result:")
print(type(result))
print(result)
