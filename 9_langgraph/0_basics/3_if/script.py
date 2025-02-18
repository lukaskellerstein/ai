from typing import Annotated, Any, Literal, Sequence
import os
import io
from dotenv import load_dotenv, find_dotenv
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from visualizer import visualize
import operator
from langgraph.graph import START, END

_ = load_dotenv(find_dotenv())  # read local .env file


# ---------------------------
# Define the graph
# ---------------------------


# State
class State(TypedDict):
    question: str
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
builder.add_node("c", TestNode("C"))
builder.add_node("d", TestNode("D"))


def route_b_or_c(state: State) -> Literal["b", "c"]:
    if state["question"].startswith("What is"):
        return ["b"]
    else:
        return ["c"]


# Add edges
builder.add_edge(START, "a")
builder.add_conditional_edges("a", route_b_or_c)
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)

# or set entry and finish points
# builder.set_entry_point("a")
# builder.set_finish_point("d")


# Graph object
graph = builder.compile()

# Visualize the graph
visualize(graph, "graph.png")


# ---------------------------
# Run the graph
# ---------------------------
result = graph.invoke({"question": "What is car?", "aggregate": []})
print("Result:")
print(result)
