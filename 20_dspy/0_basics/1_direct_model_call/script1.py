import dspy
from dspy import OllamaLocal

# model
model = OllamaLocal(model='llama2')
dspy.configure(lm=model)

# call model directly
result = model("Tell me a joke.")
print(result)

