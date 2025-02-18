import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, ToolMessage
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
    sender: str


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

    return {
        "messages": [result],
        "sender": name,
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

    return {
        "messages": [result],
        "sender": name,
    }


# ----------------------------------
# Node 3 - Tool node
# ----------------------------------
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )

            print("----- Tool Node -----")
            pprint(tool_result)

            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


ToolNode = BasicToolNode(tools=tools)


# ----------------------------------
# router edge
# ----------------------------------
def coder_router(state) -> Literal["__end__", "tool", "researcher"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]

    print("----- Coder Router -----")
    pprint(last_message)

    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "tool"

    if "[[[[FINAL ANSWER]]]]" in last_message.content:
        # Any agent decided the work is done
        return END
    else:
        return "researcher"


def researcher_router(state) -> Literal["__end__", "tool", "coder"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]

    print("----- Researcher Router -----")
    pprint(last_message)

    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "tool"

    if "[[[[FINAL ANSWER]]]]" in last_message.content:
        # Any agent decided the work is done
        return END
    else:
        return "coder"


def tool_router(state) -> Literal["coder", "researcher"]:
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    return state["sender"]


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
