from langchain.tools import YouTubeSearchTool

tool = YouTubeSearchTool()

result = tool.run("lex friedman")

print(result)
