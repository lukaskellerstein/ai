import dspy
from dspy import OllamaLocal

# model
model = OllamaLocal(model='mistral:v0.2')
dspy.configure(lm=model)

# ---------------------------------------------
# Modules
# ---------------------------------------------

# class GenerateAnswer(dspy.Signature):
#     """Answer questions with short factoid answers."""
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 1 and 5 words")

# The SIMPLEST = Predict
# qa = dspy.Predict('question -> answer')

# CoT = ChainOfThought
qa = dspy.ChainOfThought('question -> answer')

# CoT with Hint = ChainOfThoughtWithHint
# qa = dspy.ChainOfThoughtWithHint('question, hint -> answer')

# PoT = ProgramOfThought
# qa = dspy.ProgramOfThought('question -> answer')

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
