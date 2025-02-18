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
    counter: int
    # with aggregation
    data: Annotated[list, operator.add]
    # without aggregation
    # data: List[Any]


# Test node
def TestNode(name: str):
    def test_node(state: State):
        print("Test node", name)
        print(state)
        return {"data": [name]}

    return test_node


def CounterNode(state: State):
    print("Counter node")
    print(state)
    return {"counter": state["counter"] + 1}


# Build the graph
builder = StateGraph(State)
builder.add_node("a", TestNode("A"))
builder.add_node("b", TestNode("B"))
builder.add_node("c", CounterNode)
builder.add_node("d", TestNode("D"))


def shouldContinue(state: State) -> Literal["b", "d"]:
    if state["counter"] < 3:
        return ["b"]
    else:
        return ["d"]


builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", "c")
builder.add_conditional_edges("c", shouldContinue)
builder.add_edge("d", END)


# Graph object
graph = builder.compile()

# Visualize the graph
visualize(graph, "graph.png")


# ---------------------------
# Run the graph
# ---------------------------
initial_state = {"question": "What is car?", "counter": 0, "data": []}
result = graph.invoke(initial_state)
print("Result:")
print(result)
