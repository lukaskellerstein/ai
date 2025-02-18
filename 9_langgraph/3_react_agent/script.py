import operator
from typing import Annotated, Any, List, Literal
import os
import io
from dotenv import load_dotenv, find_dotenv
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from visualizer import visualize
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

_ = load_dotenv(find_dotenv())  # read local .env file


# Model
llm = ChatOpenAI(model="gpt-4-turbo")


# Tools
@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

# ---------------------------
# Define the graph
# ---------------------------

# Graph object
graph = create_react_agent(llm, tools=tools)
print("graph")
print(graph)

# Visualize the graph
visualize(graph, "graph.png")


# ---------------------------
# Run the graph
# ---------------------------
inputs = {"messages": [("user", "what is the weather in sf")]}
result = graph.invoke(inputs)
print("result")
print(result)
