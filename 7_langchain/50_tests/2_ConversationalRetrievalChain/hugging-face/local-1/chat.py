import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoConfig,
    pipeline,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

_ = load_dotenv(find_dotenv())  # read local .env file


start = time.time()

# ---------------------------
# ConversationalRetrieval Chain
#
# HAVE a Chat History !!!! (multiple questions)
# ---------------------------


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


# google/flan-t5-large > 3GB ---------------------

# try:
#     model_id = "google/flan-t5-large"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
#     pipe = pipeline(
#         "text2text-generation", model=model, tokenizer=tokenizer, max_length=500
#     )

#     local_llm = HuggingFacePipeline(pipeline=pipe)
# except Exception as e:
#     print("An exception occurred")
#     print(e)

# google/flan-t5-xl > 12GB ---------------------

try:
    model_id = "google/flan-t5-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, max_length=500
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
except Exception as e:
    print("An exception occurred")
    print(e)

# ----------------------------
# CHAIN = Option 1. - ConversationalRetrievalChain
# ----------------------------

qa = ConversationalRetrievalChain.from_llm(
    llm=local_llm, retriever=retriever, chain_type="stuff"
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
