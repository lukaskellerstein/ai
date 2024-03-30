from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# Text Completion with Chain
# Ollama - Llama2 (basic)
# ---------------------------

# ---------------------------
# prompt > model 1 
# prompt > model 2
# ---------------------------

# model
llm = Ollama(model="mistral")


# Chain 1
prompt1 = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
chain1 = prompt1 | llm

# Chain 2
prompt2 = ChatPromptTemplate.from_template("write a 2-line poem about {topic}")
chain2 = prompt2 | llm

map_chain = RunnableParallel(joke=chain1, poem=chain2)

# result
result = map_chain.invoke({"topic": "ice cream"})
print(result)
