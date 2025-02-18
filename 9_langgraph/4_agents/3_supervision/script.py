from dotenv import load_dotenv, find_dotenv
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import START, END
import json
from graph import (
    MyState,
    ResearcherNode,
    CoderNode,
    supervisor_router,
    visualize,
)
from devtools import pprint
from agent_supervisor import supervisor_chain

_ = load_dotenv(find_dotenv())  # read local .env file


# ---------------------------
# Define the graph
# ---------------------------
workflow = StateGraph(MyState)

workflow.add_node("researcher", ResearcherNode)
workflow.add_node("coder", CoderNode)
workflow.add_node("supervisor", supervisor_chain)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {"researcher": "researcher", "coder": "coder", "__end__": END},
)
workflow.add_edge("coder", "supervisor")
workflow.add_edge("researcher", "supervisor")

# Graph object
graph = workflow.compile()

# Visualize the graph
visualize(graph, "graph.png")


# ---------------------------
# Run the graph
# ---------------------------
events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Give me investment advice for MSFT stock and save it into a file,"
                "and then draw a graph of the close price and save it into a file."
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 10},
)
for s in events:
    print("=== EVENT ===")
    pprint(s)
    print("=============")
