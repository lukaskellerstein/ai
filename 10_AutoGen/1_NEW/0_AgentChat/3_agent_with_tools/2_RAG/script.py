from pprint import pprint
from typing import Annotated
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from dotenv import load_dotenv, find_dotenv
import asyncio
import os
import chromadb

load_dotenv(find_dotenv())  # read local .env file

# Initialize ChromaDB client 
client = chromadb.PersistentClient(path="./chromadb") 

# Create a collection
my_collection = collection = client.get_collection("my_collection")


llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)


# Define a tool
async def search_in_db(text: Annotated[str, "Text to search in database"]) -> str:
    results = collection.query(query_texts=[text], n_results=2)
    print("-" * 30)
    print("Results from DB")
    print(results)
    print(len(results["documents"]))
    print("-" * 30)

    docs = results["documents"][0]
    for doc in docs:
        print(doc)
        print("-" * 30)

    print("-" * 30)
    merged_docs = f"{docs[0]} \n {docs[1]}"
    return merged_docs

db_tool = FunctionTool(search_in_db, description="Search in database")

# Main function
async def main() -> None:
    agent = AssistantAgent(
        name="my_assistant", 
        model_client=llm, 
        system_message="You are a helpful AI assistant.",
        tools=[db_tool]
    )

    result = await agent.run(task="Jaké strategie jsou porovnávány mezi sebou?")
    pprint(result)
    # print(result.messages[len(result.messages) - 1].content)

# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())