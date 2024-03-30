from operator import itemgetter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import format_document
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama

# -----------------------------------------------------------------

# embedding function
embeddings = OllamaEmbeddings(model="mistral")

# load DB from disk
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Retrieval
retriever = db.as_retriever()

# model
llm = Ollama(model="mistral:v0.2") # returns TEXT
# llm = Ollama(model="mistral:instruct") # returns TEXT
# llm = ChatOllama(model="mistral:v0.2") # returns MESSAGE object
# llm = ChatOllama(model="mistral:instruct") # returns MESSAGE object

# memory
memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

# -----------------------------------------------------------------


# prompt
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

#
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

#
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)





# -----------------------------------------------------------------


# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
}
# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | llm,
    "docs": itemgetter("docs"),
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer




# -----------------------------------------------------------------

# Invoke 1
inputs = {"question": "what is Autogen?"}
result = final_chain.invoke(inputs)
print("Invoke 1")
print(result)

# Save memory
# Note that the memory does not save automatically
# This will be improved in the future
# For now you need to save it yourself
memory.save_context(inputs, {"answer": result["answer"]})


# Invoke 2
inputs = {"question": "What are the use cases for it?"}
result = final_chain.invoke(inputs)
print("Invoke 2")
print(result)