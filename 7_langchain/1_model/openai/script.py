import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

_ = load_dotenv(find_dotenv())  # read local .env file


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
        
# model
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, callbacks=[MyCustomHandler()])
# llm = ChatOpenAI(model="gpt-4", temperature=0.9, callbacks=[MyCustomHandler()])

# ----------------------

# Invoke 1
print("---- Invoke 1 ----")
result = llm.invoke(prompt)
print("Result:")
print(type(result))
print(result)

# Invoke 2 = NO HISTORY
print("---- Invoke 2 ----")
prompt = "Give me another 5."
result = llm.invoke(prompt)
print("Result:")
print(type(result))
print(result)
