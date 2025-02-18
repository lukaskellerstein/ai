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

_ = load_dotenv(find_dotenv())  # read local .env file


# Model
llm = ChatOpenAI(model="gpt-4-turbo")


# ----------------------
# Predefined tools
# ----------------------
search = TavilySearchResults(max_results=3)

# manual call of the tool
# search_result = search.invoke({"query": "What are the latest news in AI?"})
# print(search_result)

# ---------------------------
# Define the graph
# ---------------------------


# State
class State(TypedDict):
    question: str
    query: str
    search: str
    summarization: str


# Prepare query node (Chain)
def PrepareQueryNode(state: State, config: RunnableConfig):
    print("PrepareQueryNode")
    print(state)

    messages = [
        ("system", "Improve the user query, so it can be used for a better search."),
        ("human", "User question: {question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"question": state["question"]})

    return {"query": result}


# API Call node (API call)
def ApiCallNode(state: State, config: RunnableConfig):
    print("ApiCallNode")
    print(state)

    search_result = search.invoke({"query": state["query"]})
    return {"search": search_result}


# Summarize node (LLM call)
def SummarizeNode(state: State, config: RunnableConfig):
    print("SummarizeNode")
    messages = [
        SystemMessage("You are a AI assistant focused on summarizing!"),
        HumanMessage(
            f"Summarize for me this information: \n\n ===\n{state['search']}\n==="
        ),
    ]
    response = llm.invoke(messages)
    return {"summarization": response}


# Log node
def LogNode(state: State):
    print("LogNode")
    print(state)
    # return state
    pass


# Build the graph
builder = StateGraph(State)
builder.add_node("prepareQuery", PrepareQueryNode)
builder.add_node("apicall", ApiCallNode)
builder.add_node("summarize", SummarizeNode)
builder.add_node("log", LogNode)

builder.add_edge(START, "prepareQuery")
builder.add_edge("prepareQuery", "apicall")
builder.add_edge("apicall", "summarize")
builder.add_edge("summarize", "log")
builder.add_edge("log", END)

# Graph object
graph = builder.compile()

# Visualize the graph
visualize(graph, "graph.png")


# ---------------------------
# Run the graph
# ---------------------------
question = "space"
initial_state = {"question": "What are the latest news in AI?"}
result = graph.invoke(initial_state)
print("--------- Result ---------")
print(result["summarization"].content)
