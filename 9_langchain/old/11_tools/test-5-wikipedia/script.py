from langchain.utilities import WikipediaAPIWrapper

wikipedia = WikipediaAPIWrapper()

result = wikipedia.run("First Atomic Bomb")

print(result)
