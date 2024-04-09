import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.callbacks import BaseCallbackHandler

_ = load_dotenv(find_dotenv())  # read local .env file


# ----------------------
# Tools
# ----------------------
tools = [TavilySearchResults(max_results=1)]

# ----------------------
# Prompt
# ----------------------

# prompt 1
# prompt = "What would be a good company name for a company that makes colorful socks?"

# prompt 2
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

# prompt 3
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "You are AI assistant"),
#     ("human", "hi"),
#     ("ai", "hello {name}"),
#     ("human", "what is {product}?"),
# ])

# prompt = prompt_template.format_messages(
#     name="Lukas",
#     product="Langchain"
# )


print("---- Prompt ----")
print(type(prompt))
print(prompt)



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






# ----------------------
# MODEL
# = Mistral v0.2,  Mistral Instruct
# NO HISTORY !!!
# ----------------------


# LLM is not possible to use with function calls
# llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, callbacks=[MyCustomHandler()])
# add tools to the model
# llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

# Chat is possible to use with function calls
llm = ChatOpenAI(model="gpt-4", temperature=0.9, callbacks=[MyCustomHandler()])
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
prompt = "When it was founded?"
result = llm_with_tools.invoke(prompt)
print("Result:")
print(type(result))
print(result)
