from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


llm = Ollama(model="mistral")

# ---------------------------
# Routing chain = classification
# ---------------------------

chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `LangChain`, `Anthropic`, or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | llm
    | StrOutputParser()
)

chain.invoke({"question": "how do I call Anthropic?"})

# ---------------------------
# All route chains
# ---------------------------

langchain_chain = (
    PromptTemplate.from_template(
        """You are an expert in langchain. \
Always answer questions starting with "As Harrison Chase told me". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | llm
)
anthropic_chain = (
    PromptTemplate.from_template(
        """You are an expert in anthropic. \
Always answer questions starting with "As Dario Amodei told me". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | llm
)
general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question:

Question: {question}
Answer:"""
    )
    | llm
)



# ---------------------------
# ROUTER function
# ---------------------------
def route(info):
    if "anthropic" in info["topic"].lower():
        return anthropic_chain
    elif "langchain" in info["topic"].lower():
        return langchain_chain
    else:
        return general_chain


# ---------------------------
# FULL CHAIN
# ---------------------------
full_chain = {"topic": chain, "question": lambda x: x["question"]} | RunnableLambda(
    route
)

result = full_chain.invoke({"question": "how do I use Anthropic?"})
print(result)