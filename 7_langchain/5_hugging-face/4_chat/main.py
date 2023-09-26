import time
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceEndpoint

_ = load_dotenv(find_dotenv())  # read local .env file


start = time.time()


# ---------------------------
# Generic Chain
#
# Text Completion with memory
#
# ---------------------------


template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{chat_history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(
    ai_prefix="AI", human_prefix="Human", memory_key="chat_history"
)

endpoint_url = (
    "https://nwv57we1u3xj4umt.us-east-1.aws.endpoints.huggingface.cloud"
)
llm = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    huggingfacehub_api_token="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",
    task="text-generation",
)


llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)


output = llm_chain.predict(human_input="Hi there, my name is Sharon!")
print(output)

output = llm_chain.predict(
    human_input="What would be a good company name for a company that makes colorful socks?"
)
print(output)

output = llm_chain.predict(human_input="What is my name?")
print(output)

output = llm_chain.predict(human_input="Who are you in this conversation?")
print(output)


end = time.time()
print(f"NN takes: {end - start} sec.")
