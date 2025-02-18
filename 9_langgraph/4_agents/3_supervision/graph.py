import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from agent_researcher import researcher_agent
from agent_coder import coder_agent
import json
from tools import tools
import io
from PIL import Image
from devtools import pprint
from langgraph.graph import START, END


# ----------------------------------
# State
# ----------------------------------
class MyState(TypedDict):
    messages: Annotated[list, operator.add]
    # The 'next' field indicates where to route to next
    next: str


# ----------------------------------
# Node 1 = Researcher
# ----------------------------------
def ResearcherNode(state: MyState):
    """Researcher agent."""
    name = "researcher"

    print("----- Researcher Node -----")

    # Invoke the agent
    result = researcher_agent.invoke(state)
    pprint(result)

    result = HumanMessage(content=result["output"], name=name)
    return {
        "messages": [result],
    }


# ----------------------------------
# Node 2 = Coder
# ----------------------------------
def CoderNode(state: MyState):
    """Coder agent."""
    name = "coder"

    print("----- Coder Node -----")

    # Invoke the agent
    result = coder_agent.invoke(state)
    pprint(result)

    result = HumanMessage(content=result["output"], name=name)
    return {
        "messages": [result],
    }


# ----------------------------------
# router edge
# ----------------------------------
def supervisor_router(state) -> Literal["coder", "researcher", "__end__"]:
    next = state["next"]

    print("----- Supervisor Router -----")
    pprint(next)

    return next


# ---------------------------
# Visualize the graph
# ---------------------------
def visualize(graph, output_file_name):
    try:
        png = graph.get_graph().draw_mermaid_png()
        # Create a Pillow Image object from the image data
        pil_image = Image.open(
            io.BytesIO(png)
        )  # Replace io.BytesIO with appropriate stream if necessary

        # Save the image to a file
        output_file = output_file_name  # Replace with your desired output file path
        pil_image.save(output_file, "PNG")
    except Exception:
        # This requires some extra dependencies and is optional
        pass
