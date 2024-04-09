import dspy
from dspy import OllamaLocal

# model
model = OllamaLocal(model='mistral:v0.2')
dspy.configure(lm=model)


# ------------------------------------------
# call model directly
# ------------------------------------------
print("---- Direct model call ----")
result = model("Tell me a joke.")
print(result)

# ------------------------------------------
# call model as DSPy
# ------------------------------------------
print("---- DSPy call ----")
qa = dspy.Predict('input -> output')
print(qa.signature)

input = "Tell me a joke."
response = qa(input=input)
print(f"Input: {input}")
print(f"Output: {response.output}")
