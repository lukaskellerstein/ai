from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# ----------------------
# Simple Prompt = just a string
# ----------------------

# prompt
prompt = "What is a good name for a company that makes colorful socks?"
print("---- Prompt ----")
print(type(prompt))
print(prompt)

# ----------------------
# Prompt template 1 = Instance of class
# ----------------------

# prompt template
promptemplate = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# prompt
prompt = promptemplate.format(product="colorful socks")
print("---- Prompt ----")
print(type(prompt))
print(prompt)

# ----------------------
# Prompt template 2 = "from_template" method
# ----------------------

# prompt template
template = "What is a good name for a company that makes {product}?"
promp_template = PromptTemplate.from_template(template)

# prompt
prompt = promp_template.format(product="colorful socks")
print("---- Prompt ----")
print(type(prompt))
print(prompt)

# ----------------------
# Chat template 1 = "from_template" method
# (allows creating a template for a list of chat messages)
# ----------------------

# prompt template
template = "What is a good name for a company that makes {product}?"
prompt_template = ChatPromptTemplate.from_template(template)

# prompt
prompt = promp_template.format(product="colorful socks")
print("---- Prompt ----")
print(type(prompt))
print(prompt)


# ----------------------
# Chat template 2 = "from_messages" method
# (allows creating a template for a list of chat messages)
# ----------------------

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are AI assistant"),
        ("human", "hi {name}"),
        ("ai", "hello"),
        ("human", "What is a good name for a company that makes {product}?"),
    ]
)

prompt = prompt_template.format_messages(name="Lukas", product="colorful socks")
print("---- Prompt ----")
print(type(prompt))
print(prompt)


# ----------------------
# ???
# ----------------------
prompt = [
    {
        "role": "system",
        "content": "You are a helpful assistant with output as Shakespear.",
    },
    {
        "role": "user",
        "content": "What would be a good company name for a company that makes colorful socks?",
    },
]

print("---- Prompt ----")
print(type(prompt))
print(prompt)
