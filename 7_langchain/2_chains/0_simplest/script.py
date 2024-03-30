from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# Text Completion with Chain
# Ollama - Llama2 (basic)
# ---------------------------

# ---------------------------
# prompt > model 
# ---------------------------

# prompt
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")

# model
llm = Ollama(model="mistral")


# Chain
chain = prompt | llm

# result
result = chain.invoke({"topic": "ice cream"})
print(result)


# ---------------------------
# prompt > model > output parser
# ---------------------------

# output parser
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

# result
result = chain.invoke({"topic": "ice cream"})
print(result)
