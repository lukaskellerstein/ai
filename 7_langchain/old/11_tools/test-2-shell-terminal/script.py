from langchain.tools import ShellTool

shell_tool = ShellTool()

result = shell_tool.run({"commands": ["echo 'Hello World!'", "time"]})
print(result)
