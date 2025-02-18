import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint

_ = load_dotenv(find_dotenv())  # read local .env file

# ----------------------
# Hugging face Hub
# ----------------------

# prompt
prompt = "What would be a good company name for a company that makes colorful socks?"

# model
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.9}
)
result = llm.invoke(prompt)
print(result)

# ----------------------
# Hugging face Endpoint
# ----------------------

repo_id = "https://fayjubiy2xqn36z0.us-east-1.aws.endpoints.huggingface.cloud"

llm = HuggingFaceEndpoint(
    endpoint_url=f"{repo_id}",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)
result = llm.invoke(prompt)
print(result)