from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

_ = load_dotenv(find_dotenv())  # read local .env file


# ---------------------------
# Text Completion in a Chain with Prompt Template
# ---------------------------

llm = HuggingFaceHub(
    repo_id="facebook/blenderbot-1B-distill",
    model_kwargs={"temperature": 0, "max_length": 64},
)

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
)

result = chain.run("colorful socks")
print(result)

result = chain.run("jet engine cars")
print(result)
