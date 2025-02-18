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
from langgraph.graph import START, END

_ = load_dotenv(find_dotenv())  # read local .env file


# Model
llm = ChatOpenAI(model="gpt-4-turbo")


# ---------------------------
# Define the graph
# ---------------------------


# State
class State(TypedDict):
    theme: str
    jokeText: str
    storyText: str


# Joke node
def JokeNode(state: State):
    messages = [
        SystemMessage("You are a hiliarious person!"),
        HumanMessage(f"Generate a joke on theme: {state['theme']}"),
    ]
    response = llm.invoke(messages)
    return {"jokeText": response}


# Story node
def StoryNode(state: State):
    messages = [
        SystemMessage("You are a storyteller!"),
        HumanMessage(f"Generate a story for theme: {state['theme']}"),
    ]
    response = llm.invoke(messages)
    return {"storyText": response}


# Build the graph
builder = StateGraph(State)
builder.add_node("joke", JokeNode)
builder.add_node("story", StoryNode)

builder.add_edge(START, "joke")
builder.add_edge("joke", "story")
builder.add_edge("story", END)

# Graph object
graph = builder.compile()

# Visualize the graph
visualize(graph, "graph.png")


# ---------------------------
# Run the graph
# ---------------------------
theme = "space"
initial_state = {"jokeText": "", "storyText": "", "theme": theme}
result = graph.invoke(initial_state)
print(result)
