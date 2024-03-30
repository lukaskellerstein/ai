from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ----------------------
# Text Completion = Mistral v0.2
# WITH HISTORY !!!
# ----------------------

# model
# llm = Ollama(model="mistral:v0.2") # returns TEXT
# llm = Ollama(model="mistral:instruct") # returns TEXT
# llm = ChatOllama(model="mistral:v0.2") # returns MESSAGE object
llm = ChatOllama(model="mistral:instruct") # returns MESSAGE object

# ----------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# result = llm.invoke(prompt)
# print("---- Answer 3 ----")
# print(result)

chain = prompt | llm

# history
chat_history = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Invoke 1
prompt = "What would be a good company name for a company that makes colorful socks?"
result = chain_with_history.invoke(
    {"input": prompt},
    {"configurable": {"session_id": "unused"}},
)
print("---- Answer 1 ----")
print(type(result))
print(result)

# Invoke 2
prompt = "Give me another 5."
result = chain_with_history.invoke(
    {"input": prompt},
    {"configurable": {"session_id": "unused"}},
)
print("---- Answer 2 ----")
print(type(result))
print(result)


# + possible modifying history = alternating chat scenario
# + keep only last 10 messages = trim messages for small context window
# + summarize history = summarize history for small context window