from langchain_community.embeddings import OllamaEmbeddings

# model
llm = OllamaEmbeddings(model="mistral")

result = llm.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!",
    ]
)

print(result)
