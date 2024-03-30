from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate

# ----------------------
# Text Completion 
# = Mistral v0.2,  Mistral Instruct
# NO HISTORY !!!
# ----------------------

# model
# llm = Ollama(model="mistral:v0.2") # returns TEXT
# llm = Ollama(model="mistral:instruct") # returns TEXT
# llm = ChatOllama(model="mistral:v0.2") # returns MESSAGE object
llm = ChatOllama(model="mistral:instruct") # returns MESSAGE object

# ----------------------

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

# ----------------------
# Homework: Why can I call also array of messages on llm? Why we have then ChatModels?

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
