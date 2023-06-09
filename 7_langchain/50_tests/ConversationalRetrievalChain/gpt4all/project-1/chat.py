import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from chromadb.config import Settings
from langchain.llms import GPT4All
from langchain.embeddings import HuggingFaceEmbeddings

_ = load_dotenv(find_dotenv())  # read local .env file


start = time.time()


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "db"
chroma_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory,
    anonymized_telemetry=False,
)
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    client_settings=chroma_settings,
)
retriever = db.as_retriever(search_kwargs={"k": 1})


llm = GPT4All(
    model="./model/ggml-gpt4all-j-v1.3-groovy.bin",
    n_ctx=1000,
    backend="gptj",
    verbose=False,
)
qa = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, chain_type="stuff"
)
# qa = RetrievalQA.from_chain_type(
#     llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
# )

questions = [
    # "What is this app about?",
    # "What assets are covered in the app?",
    "Is in the code use async/await?",
    "What is the most efficient class in the code?",
    "What is the class hierarchy?",
    "What classes are derived from the DBObject class?",
]
chat_history = []

for question in questions:
    print(f"-> **Question**: {question} \n")
    result = qa({"query": question, "chat_history": chat_history})
    chat_history.append((question, result["result"]))
    print(f"**Answer**: {result['result']} \n")


end = time.time()
print(f"NN takes: {end - start} sec.")
