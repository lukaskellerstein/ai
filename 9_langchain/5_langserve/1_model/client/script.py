from langchain.prompts.chat import ChatPromptTemplate
from langserve import RemoteRunnable

# model
llm = RemoteRunnable("http://localhost:8000/ollama/")

# prompt
prompt = "Tell me a 3 sentence story about a cat."

# -----------------------------------------------------------------

# Invoke 1
prompt = "What would be a good company name for a company that makes colorful socks?"
result = llm.invoke(prompt)
print("---- Answer 1 ----")
print(type(result))
print(result)

# Invoke 2
prompt = "Give me another 5."
result = llm.invoke(prompt)
print("---- Answer 2 ----")
print(type(result))
print(result)


# Invoke 3 = Array of messages
messages = [
    ("system", "You are AI assistant"),
    ("human", "hi {name}"),
    ("ai", "hello"),
    ("human", "What is a good name for a company that makes {product}?"),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

messages = prompt_template.format_messages(
    name="Lukas",
    product="colorful socks"
)

result = llm.invoke(messages)
print("---- Answer 3 ----")
print(type(result))
print(result)