from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# Text Completion with Chain
# Ollama - Llama2 (basic)
# ---------------------------

llm = Ollama(model="llama2")

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


print(chain.invoke({"topic":"ice cream" }))
