import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain

_ = load_dotenv(find_dotenv())  # read local .env file


start = time.time()


# ----------------------------
# DATA
# ----------------------------

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
retriever = db.as_retriever()

# ----------------------------
# LLM
# ----------------------------

# LIMIT MAX 1000 TOKENS ==> NOT GOOD

llm = HuggingFaceHub(repo_id="bigscience/bloom")

# ----------------------------
# CHAIN = Option 1. - ConversationalRetrievalChain
# ----------------------------

qa = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, chain_type="stuff"
)

questions = [
    "What is this app about?",
    "What Assets are used in the app?",
    "Is in the code use async/await?",
    "What is the most efficient class in the code?",
    "What is the class hierarchy?",
    "What classes are derived from the DBObject class?",
]
chat_history = []

for question in questions:
    print(f"Question: {question} \n")
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    print(f"Answer: {result['answer']} \n")
    print(" ")


end = time.time()
print(f"NN takes: {end - start} sec.")
