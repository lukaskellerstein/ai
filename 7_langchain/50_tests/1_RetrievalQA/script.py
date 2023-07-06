import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI
from langchain.chains import RetrievalQA

_ = load_dotenv(find_dotenv())  # read local .env file


start = time.time()


# ---------------------------
# RetrievalQA Chain
#
# No Chat History !!! (only one question)
# ---------------------------


embeddings = OpenAIEmbeddings()

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

llm = OpenAI(temperature=0)

# ----------------------------
# Option 1. - RetrievalQA Chain
# ----------------------------


qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

questions = [
    "Who is Forrest Gump?",
    "Where was born Forrest?",
    "Were Forrest Gump born in Alabama?",
    "What is a Gotham city?",
    "What is a black hole?",
    "What is time?",
    "What is the first question that I asked?",
]

for question in questions:
    print(f"Question: {question} \n")
    result = qa.run(question)
    print(f"Answer: {result} \n")
    print(" ")


end = time.time()
print(f"NN takes: {end - start} sec.")
