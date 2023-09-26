import time
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceEndpoint

_ = load_dotenv(find_dotenv())  # read local .env file


start = time.time()

# ---------------------------
# Text Completion in a Chain with Prompt Template
# ---------------------------

endpoint_url = (
    "https://nwv57we1u3xj4umt.us-east-1.aws.endpoints.huggingface.cloud"
)
llm = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    huggingfacehub_api_token="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",
    task="text-generation",
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


end = time.time()
print(f"NN takes: {end - start} sec.")
