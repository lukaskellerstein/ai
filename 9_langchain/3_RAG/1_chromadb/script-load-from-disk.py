from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# embeddings
embeddings = OllamaEmbeddings(model="mistral")

# load DB from disk
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Similarity search
docs = db.similarity_search("What is Autogen?")
print(docs)