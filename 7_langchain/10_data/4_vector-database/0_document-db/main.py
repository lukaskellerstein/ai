import os
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

_ = load_dotenv(find_dotenv())  # read local .env file

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

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)


qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="map_reduce", retriever=docsearch.as_retriever()
)

query = "Who was killed first by Joker?"
result = qa.run(query)
print(result)

query = "Can you summarize Tenet in five sentences?"
result = qa.run(query)
print(result)
