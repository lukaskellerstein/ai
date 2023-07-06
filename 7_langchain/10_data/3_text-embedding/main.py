import os
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())  # read local .env file

embeddings = OpenAIEmbeddings()

result = embeddings.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!",
    ]
)

print(result)
