import dspy
from dspy import OllamaLocal

# model
model = OllamaLocal(model='mistral:v0.2')
dspy.configure(lm=model)

# ---------------------------------------------
# Custom Signature = Predict
# ---------------------------------------------

# Custom: input -> output
class CustomQA(dspy.Signature):
    # """Generate an output given a input in style of Einstein."""
    """Classify input among ['question', 'answer', 'greetings', 'command', 'statement', 'opinion', 'insult']"""
    
    input = dspy.InputField()
    output = dspy.OutputField()

qa = dspy.Predict(CustomQA)

print("-------------------------------------------------------------")

print("---- QA full object ----")
print(qa)

print("---- Signature ----")
print(qa.signature)

print("-------------------------------------------------------------")


# Run with the default LM configured with `dspy.configure` above.
# input = "How many floors are in the castle David Gregory inherited?"
# input = "Hello"
# input = "I think the car should be blue."
input = "No, you are wrong."
response = qa(input=input)

print("---- input ----")
print(input)

print("---- output ----")
print(response.output)

print("---- Full Response from model ----")
print(response)

