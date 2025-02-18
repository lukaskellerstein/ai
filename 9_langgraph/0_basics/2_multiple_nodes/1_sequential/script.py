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
from langgraph.graph import START, END

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------
# Define the graph
# ---------------------------


# State
class State(TypedDict):
    # with aggregation
    data: Annotated[list, operator.add]
    # without aggregation
    # data: List[Any]


# Test node
def TestNode(name: str):
    def test_node(state: State):
        return {"data": [name]}

    return test_node


# Build the graph
builder = StateGraph(State)
builder.add_node("a", TestNode("A"))
builder.add_node("b", TestNode("B"))

# Add edges
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

# or set entry and finish points
# builder.set_entry_point("a")
# builder.set_finish_point("b")

# Graph object
graph = builder.compile()

# Visualize the graph
visualize(graph, "graph.png")


# ---------------------------
# Run the graph
# ---------------------------
initial_state = {"data": []}
result = graph.invoke(initial_state)
print(result)
