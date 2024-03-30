from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel
from operator import itemgetter


# -----------------------------------------------------------------

# embedding function
embeddings = OllamaEmbeddings(model="mistral")

# load DB from disk
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Retrieval
retriever = db.as_retriever()

# model
model = Ollama(model="mistral")


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



# Chain 1
_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser(),
)


# Chain 2
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

# FULL Chain
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | model


# -----------------------------------------------------------------

# Invoke 1
# result = conversational_qa_chain.invoke(
#     {
#         "question": "What is Autogen?",
#         "chat_history": [],
#     }
# )
# print(result)

# Invoke 2
result = conversational_qa_chain.invoke(
    {
        "question": "What are the use cases for it?",
        "chat_history": [
            HumanMessage(content="What is the best AI Agents framework?"),
            AIMessage(content="Autogen."),
        ],
    }
)
print(result)

