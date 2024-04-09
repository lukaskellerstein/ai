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

# Test Retreiver
# results = retriever_model("Who is Forrest?", k=5)
# for result in results:
#     print("Document:", result.long_text, "\n")


# ---------------------------------------------
# Local model
# ---------------------------------------------
model = OllamaLocal(model='mistral:v0.2')

# ---------------------------------------------
# DSPy configure
# ---------------------------------------------
dspy.configure(lm=model, rm=retriever_model)

# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------

# Signature
# class GenerateAnswer(dspy.Signature):
#     """Answer questions with short factoid answers."""

#     context = dspy.InputField(desc="may contain relevant facts")
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 1 and 5 words")

# Custom module
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("question, context -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


# RAG = Retreival Augmented Generation
qa = RAG()

print("-------------------------------------------------------------")

question = "Who is Forrest?"
response = qa(question)

print("---- Question ----")
print(question)

print("---- Answer ----")
print(response.answer)

print("---- Full Response from model ----")
print(response)


print("--------------------------")
print("---- HISTORY OF STEPS ----")
print("--------------------------")
model.inspect_history(n=10)
