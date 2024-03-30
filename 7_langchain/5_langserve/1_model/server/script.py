from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

# model
llm = ChatOllama(model="mistral:v0.2")

add_routes(
    app,
    llm,
    path="/ollama",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)