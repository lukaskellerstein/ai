import dspy
from dspy import OllamaLocal
from dspy.retrieve.chromadb_rm import ChromadbRM
from chromadb.utils import embedding_functions

# ---------------------------------------------
# Retreival
# ---------------------------------------------

# embeding
embedding = embedding_functions.DefaultEmbeddingFunction()

retriever_model = ChromadbRM(
    'my_collection',
    './chromadb',
    embedding_function=embedding,
    k=5
)

# model
model = OllamaLocal(model='mistral:v0.2')
dspy.configure(lm=model, rm=retriever_model)


# ---------------------------------------------
# Modules
# ---------------------------------------------

# ReAct = Reason and Act
qa = dspy.ReAct('question -> answer', tools=[])

print("-------------------------------------------------------------")

question = "Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?"
# hint = "Parallel drying is possible."
response = qa(question=question)

print("---- Question ----")
print(question)

print("---- Rational (Chain Of Thoughts) ----")
print(response.rationale)

print("---- Answer ----")
print(response.answer)

# print("---- Full Response from model ----")
# print(response)


print("--------------------------")
print("---- HISTORY OF STEPS ----")
print("--------------------------")
model.inspect_history(n=10)
