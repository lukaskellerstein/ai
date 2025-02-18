from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ---------------------------
# Text Completion with Chain
# ---------------------------

YOUR_POD_ID = "3zynknt1m7tspe"
YOUT_POD_PORT = 8080
API_URL = f"https://{YOUR_POD_ID}-{YOUT_POD_PORT}.proxy.runpod.net"


# ---------------------------
# prompt > model
# ---------------------------

# prompt
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")

# model
# llm = Ollama(model="mistral")
llm = ChatOpenAI(
    model="tgi",
    base_url=f"https://{API_URL}/v1",
)

# Chain
chain = prompt | llm

# result
result = chain.invoke({"topic": "ice cream"})
print(result)


# ---------------------------
# prompt > model > output parser
# ---------------------------

# output parser
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

# result
result = chain.invoke({"topic": "ice cream"})
print(result)
