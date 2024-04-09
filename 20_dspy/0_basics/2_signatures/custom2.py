import dspy
from dspy import OllamaLocal

# model
model = OllamaLocal(model='mistral:v0.2')
dspy.configure(lm=model)

# ---------------------------------------------
# Custom Signature = Predict
# ---------------------------------------------

# Custom: input, context -> output
class CustomQA(dspy.Signature):
    """Answer a question given a context."""
    
    input = dspy.InputField()
    context = dspy.InputField()
    output = dspy.OutputField()

qa = dspy.Predict(CustomQA)

print("-------------------------------------------------------------")

print("---- QA full object ----")
print(qa)

print("---- Signature ----")
print(qa.signature)

print("-------------------------------------------------------------")


input = "Explain autogen for 5 year old."
context = "AutoGen2 is an open-source framework that allows developers to build LLM ap- plications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, conversable, and can operate in vari- ous modes that employ combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. AutoGen serves as a generic framework for building diverse applications of various complexities and LLM capacities. Em- pirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answer- ing, operations research, online decision-making, entertainment, etc."
response = qa(input=input, context=context)

print("---- input ----")
print(input)

print("---- output ----")
print(response.output)

print("---- Full Response from model ----")
print(response)

