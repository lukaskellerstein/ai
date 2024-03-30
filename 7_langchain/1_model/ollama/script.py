import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import Ollama

_ = load_dotenv(find_dotenv())  # read local .env file

# ----------------------
# Text Completion
# LLM
# ----------------------

# prompt
prompt = "What would be a good company name for a company that makes colorful socks?"

# model
llm = Ollama(model="mistral", temperature=0.9)
result = llm.invoke(prompt)
print(result)
