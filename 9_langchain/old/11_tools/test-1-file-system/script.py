from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)

from langchain.agents.agent_toolkits import FileManagementToolkit

toolkit = FileManagementToolkit(
    root_dir=str("./test-1-file-system"),
    selected_tools=["read_file", "write_file", "list_directory"],
)
tools = toolkit.get_tools()


# CopyFileTool
# DeleteFileTool
# FileSearchTool
# MoveFileTool
# ReadFileTool
# WriteFileTool
# ListDirectoryTool

print(tools)

read_tool, write_tool, list_tool = tools
write_tool.run({"file_path": "example.txt", "text": "Hello World!"})
