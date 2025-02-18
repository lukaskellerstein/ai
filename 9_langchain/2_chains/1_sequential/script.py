from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# prompt > model 1 > model 2 > output
# ---------------------------

# model
llm = Ollama(model="mistral")


# Chain 1
prompt1 = ChatPromptTemplate.from_template("what is the city {person} is from? Return name of the city and nothing else.")
chain1 = prompt1 | llm | StrOutputParser()

# Chain 2
prompt2 = ChatPromptTemplate.from_template("what country is the city {city} in? Return name of the country and nothing else.")
chain2 = (
    {"city": chain1}
    | prompt2
    | llm
    | StrOutputParser()
)

# Chain 3
prompt3 = ChatPromptTemplate.from_template("what continent is the country {country} in? Return name of the continent and nothing else.")
chain3 = (
    {"country": chain2}
    | prompt3
    | llm
    | StrOutputParser()
)

# result
result = chain3.invoke({"person": "obama"})
print(result)
