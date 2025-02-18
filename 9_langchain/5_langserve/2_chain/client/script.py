from langchain.prompts.chat import ChatPromptTemplate
from langserve import RemoteRunnable

# model
llm = RemoteRunnable("http://localhost:8000/ollama/")

# -----------------------------------------------------------------

# Invoke 1
prompt = {"product": "colorful socks"}
result = llm.invoke(prompt)
print("---- Answer 1 ----")
print(type(result))
print(result)

# Invoke 2
prompt = {"product": "car brand"}
result = llm.invoke(prompt)
print("---- Answer 2 ----")
print(type(result))
print(result)
