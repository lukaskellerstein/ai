import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
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
texts = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="mistral")

# save DB to disk
db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
