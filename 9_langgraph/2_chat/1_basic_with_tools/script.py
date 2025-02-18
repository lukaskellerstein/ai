from typing import Annotated
import os
import io
from dotenv import load_dotenv, find_dotenv
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from PIL import Image
from langchain_community.tools.tavily_search import TavilySearchResults
import json
from langchain_core.messages import ToolMessage
from typing import Literal

_ = load_dotenv(find_dotenv())  # read local .env file


# Model
llm = ChatOpenAI(model="gpt-4-turbo")

# Tools
tool = TavilySearchResults(max_results=2)
tools = [tool]

# test the tool
# tool.invoke("What's a 'node' in LangGraph?")

# add tools to the model
llm = llm.bind_tools(tools)

# ---------------------------
# Define the graph
# ---------------------------

# State
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Node 1 ----------
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Node 2 ----------

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
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])


# Edge 1 -------------------

def route_tools(
    state: State,
) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"




# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", "__end__": "__end__"},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")






# Entry and finish points
graph_builder.set_entry_point("chatbot")


# Graph object
graph = graph_builder.compile()

# ---------------------------
# Visualize the graph
# ---------------------------
try:
    png = graph.get_graph().draw_mermaid_png()
    # Create a Pillow Image object from the image data
    pil_image = Image.open(io.BytesIO(png))  # Replace io.BytesIO with appropriate stream if necessary

    # Save the image to a file
    output_file = 'output_image.png'  # Replace with your desired output file path
    pil_image.save(output_file, 'PNG')
except Exception:
    # This requires some extra dependencies and is optional
    pass


# ---------------------------
# Run the graph
# ---------------------------
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)



# ---------------------------
# UI
# ---------------------------
# import sys
# from PyQt6.QtGui import QPixmap
# from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel

# class MainWindow(QMainWindow):

#     def __init__(self):
#         super(MainWindow, self).__init__()
#         self.title = "Image Viewer"
#         self.setWindowTitle(self.title)

#         label = QLabel(self)
#         pixmap = QPixmap('output_image.png' )
#         label.setPixmap(pixmap)
#         self.setCentralWidget(label)
#         self.resize(pixmap.width(), pixmap.height())


# app = QApplication(sys.argv)
# w = MainWindow()
# w.show()
# sys.exit(app.exec())