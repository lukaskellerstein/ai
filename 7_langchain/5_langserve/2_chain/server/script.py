from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama
from langserve import add_routes
from langchain.prompts import PromptTemplate

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

# model
llm = ChatOllama(model="mistral:v0.2")

# prompt
promp_template = PromptTemplate.from_template(
    "What is a good name for a company that makes {product}?."
)

chain = promp_template | llm

add_routes(
    app,
    chain,
    path="/ollama",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)