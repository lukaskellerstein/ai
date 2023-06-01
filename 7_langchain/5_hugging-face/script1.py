from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

_ = load_dotenv(find_dotenv())  # read local .env file


template = """Question: {question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.1, "max_length": 64}
)

llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
)


question = "What is the capital of France?"

print(llm_chain.run(question))

question = "What area is best for growing wine in France?"

print(llm_chain.run(question))
