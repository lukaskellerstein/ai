from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

embeddings = OllamaEmbeddings(model="mistral")

# load DB from disk
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Retrieval
retriever = db.as_retriever()

# prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# model
llm = Ollama(model="mistral:v0.2") # returns TEXT
# llm = Ollama(model="mistral:instruct") # returns TEXT
# llm = ChatOllama(model="mistral:v0.2") # returns MESSAGE object
# llm = ChatOllama(model="mistral:instruct") # returns MESSAGE object

# chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result = chain.invoke({"question": "What is Autogen?"})
print(result)