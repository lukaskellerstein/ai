import dspy
from dspy import OllamaLocal

# model
model = OllamaLocal(model='mistral:v0.2')
dspy.configure(lm=model)

# ---------------------------------------------
# Basic Signature = Predict
# ---------------------------------------------

# Q&A = question -> answer 
# Sentiment = sentence -> sentiment <-------------------------
# Summarization = document -> summary

# Basic Signature
qa = dspy.Predict('sentence -> sentiment')
# qa = dspy.Predict('input -> output')
# qa = dspy.Predict('input -> emotion')
# qa = dspy.Predict('input -> sentiment')
# qa = dspy.Predict('input -> mood')

print("-------------------------------------------------------------")

print("---- QA full object ----")
print(qa)

print("---- Signature ----")
print(qa.signature)

print("-------------------------------------------------------------")


sentence = "it's a charming and often affecting journey."
response = qa(sentence=sentence)

print("---- sentence ----")
print(sentence)

print("---- Sentiment ----")
print(response.sentiment)

print("---- Full Response from model ----")
print(response)

