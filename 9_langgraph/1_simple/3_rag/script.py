

import operator
from typing import Annotated, Any, List
import os
import io
from dotenv import load_dotenv, find_dotenv
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from visualizer import visualize
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())  # read local .env file



# Model
llm = ChatOpenAI(model="gpt-4-turbo")


# load DB from disk
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


# ---------------------------
# Define the graph
# ---------------------------

# State
class State(TypedDict):
    question: str
    query: str
    docs: str
    answer: str


# Prepare query node (Chain)
def PrepareQueryNode(state: State, config: RunnableConfig):
    print("PrepareQueryNode")
    print(state)

    messages = [
        ("system", "Improve the user query, so it can be used for a query in vector DB."),
        ("human", "User question: {question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"question": state['question']})

    return {"query": result}

# DB Call node 
def GetDataFromDBNode(state: State, config: RunnableConfig):
    print("GetDataFromDBNode")
    print(state)

    docs = db.similarity_search(state["query"])

    return {"docs": docs}

# Format answer node (LLM call)
def FormatAnswerNode(state: State, config: RunnableConfig):
    print("FormatAnswerNode")
    print(state)

    messages = [
        SystemMessage("You are a AI assistant focused on formatting the clear result answer from question and context!"),
        HumanMessage(f"""
                     Question: 
                     ===
                     {state['question']}
                     ===
                     
                     Context:
                     ===
                     {state['docs']}
                     ===
                     """),
    ]
    response = llm.invoke(messages)
    return {"answer": response}

# Log node 
def LogNode(state: State):
    print("LogNode")
    print(state)
    return state

# Build the graph
builder = StateGraph(State)
builder.add_node("prepareQuery", PrepareQueryNode)
builder.add_node("calldb", GetDataFromDBNode)
builder.add_node("formatAnswer", FormatAnswerNode)
builder.add_node("log", LogNode)

builder.add_edge(START, "prepareQuery")
builder.add_edge("prepareQuery", "calldb")
builder.add_edge("calldb", "formatAnswer")
builder.add_edge("formatAnswer", "log")
builder.add_edge("log", END)

# Graph object
graph = builder.compile()

# Visualize the graph
visualize(graph, "graph.png")


# ---------------------------
# Run the graph
# ---------------------------
question = "space"
initial_state = { "question": "What are use cases for Autogen?"}
result = graph.invoke(initial_state)
print("--------- Result ---------")
print(result["answer"].content)