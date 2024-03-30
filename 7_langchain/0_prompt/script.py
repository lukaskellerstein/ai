from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# ----------------------
# Simple Prompt
# ----------------------

# prompt
prompt = "What would be a good company name for a company that makes colorful socks?"


# ----------------------
# Prompt template
# ----------------------

# prompt template
promptemplate = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# prompt
prompt = promptemplate.format(product="colorful socks")


# ----------------------
# Chat template 
# ----------------------

# FINISH !!!

# prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# ----------------------
# Chat template + Prompt template
# ----------------------

# chat template
chatTemplate = (
    SystemMessage(content="You are AI assistant") 
    + HumanMessage(content="hi") 
    + AIMessage(content="hello") 
    + "{input}"
)

# prompt template
promptemplate = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# prompt
prompt = promptemplate.format(product="colorful socks")

# chat prompt
chatPrompt = chatTemplate.format_messages(input=prompt)

