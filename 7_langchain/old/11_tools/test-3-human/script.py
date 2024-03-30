from langchain.callbacks.human import HumanApprovalCallbackHandler
from langchain.tools import ShellTool

tool = ShellTool(callbacks=[HumanApprovalCallbackHandler()])

print(tool.run("ls /usr"))

print(tool.run("ls /private"))
