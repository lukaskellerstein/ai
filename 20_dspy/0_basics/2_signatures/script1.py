import dspy
from dspy import OllamaLocal

# model
model = OllamaLocal(model='mistral:v0.2')
dspy.configure(lm=model)

# ---------------------------------------------
# Basic Signature = Predict
# ---------------------------------------------

# Q&A = question -> answer <-------------------------
# Sentiment = sentence -> sentiment
# Summarization = document -> summary

# Basic Signature
qa = dspy.Predict('question -> answer')
# qa = dspy.Predict('input -> output')
# qa = dspy.Predict('request -> response')

print("-------------------------------------------------------------")

print("---- QA full object ----")
print(qa)

print("---- Signature ----")
print(qa.signature)

print("-------------------------------------------------------------")

question = "If we lay 5 shirts out in the sun, it takes 4 hours for the shirts to dry. How long does it take to dry 20 shirt?"
response = qa(question=question)

print("---- Question ----")
print(question)

print("---- Answer ----")
print(response.answer)

print("---- Full Response from model ----")
print(response)

