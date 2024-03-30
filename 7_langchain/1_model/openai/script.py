import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI, ChatOpenAI

_ = load_dotenv(find_dotenv())  # read local .env file

# ----------------------
# Text Completion
# LLM
# ----------------------

# prompt
prompt = "What would be a good company name for a company that makes colorful socks?"

# model
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)
result = llm.invoke(prompt)
print(result)

# ----------------------
# Chat
# ----------------------

# prompt
prompt = "What would be a good company name for a company that makes colorful socks?"

# model
chat = ChatOpenAI(model="gpt-4", temperature=0.9)
result = chat.invoke(prompt)
print(result)