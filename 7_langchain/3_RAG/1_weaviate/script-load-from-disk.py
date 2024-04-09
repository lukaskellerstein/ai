from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import OllamaEmbeddings
import weaviate

# embeddings
embeddings = OllamaEmbeddings(model="mistral:v0.2")

# connect to DB
client = weaviate.Client("http://127.0.0.1:8080")
db = Weaviate(client, "MyDocument", "text")

# Similarity search
docs = db.similarity_search("What is Autogen?", k=5)
print(docs)