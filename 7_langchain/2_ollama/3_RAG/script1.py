import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # -------------------------------------
# # -------------------------------------
# # 1. Fill DB
# # -------------------------------------
# # -------------------------------------

# root_dir = "./source"

# docs = []

# for dirpath, dirnames, filenames in os.walk(root_dir):
#     for file in filenames:
#         if file.endswith(".txt"):
#             try:
#                 loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
#                 docs.extend(loader.load_and_split())
#             except Exception as e:
#                 pass

# print(f"{len(docs)}")


# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(docs)

# # embeddings = HuggingFaceEmbeddings()
# embeddings = OllamaEmbeddings()
# db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")


# -------------------------------------
# -------------------------------------
# 2. Langchain
# -------------------------------------
# -------------------------------------
embeddings = OllamaEmbeddings()

# db
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db.as_retriever()

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Model
model = Ollama(model="mixtral")

# Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


# -------------------------------------
# -------------------------------------
# Usage
# -------------------------------------
# -------------------------------------

print(chain.invoke({"question": """
                    For each movie, create a set of 20 questions and answers, based on the storyline and characters. 
                    Each question should be unique and not a simple factoid.
                    All question and answer pairs should cover the movie well.
                    All questions and aswers will be used to fine-tune a language model, so they should be diverse and cover the movie well.
                    Format the questions and answers as follows:
                    ===
                    {"messages": [{"role": "user", "content": "<HERE PUT QUESTION>"}, {"role": "assistant", "content": "<HERE PUT ANSWER>"}]}
                    ===
                    """ }))

#In the movie Interstellar, the robot is named TARS (Tactical Automated Response System). TARS is a helpful and informative robot that assists the astronauts on their journey through space.
# print(chain.invoke({"question": "Who is TARS and what is his mission?" }))
