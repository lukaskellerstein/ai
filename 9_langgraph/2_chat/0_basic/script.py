from typing import Annotated
from dotenv import load_dotenv, find_dotenv
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from visualizer import visualize
from langgraph.graph import START, END


_ = load_dotenv(find_dotenv())  # read local .env file


# Model
llm = ChatOpenAI(model="gpt-4-turbo")


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


# Node 1
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

# Entry and finish points
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# Graph object
graph = graph_builder.compile()

# Visualize the graph
visualize(graph, "graph.png")


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
#         pixmap = QPixmap('graph.png' )
#         label.setPixmap(pixmap)
#         self.setCentralWidget(label)
#         self.resize(pixmap.width(), pixmap.height())


# app = QApplication(sys.argv)
# w = MainWindow()
# w.show()
# sys.exit(app.exec())
