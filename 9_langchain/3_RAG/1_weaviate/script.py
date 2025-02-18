from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os
from langchain_community.embeddings import OllamaEmbeddings


root_dir = "./source"

docs = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass

print(f"{len(docs)}")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="mistral:v0.2")

db = Weaviate.from_documents(docs, embeddings, weaviate_url=WEAVIATE_URL, by_text=False)
