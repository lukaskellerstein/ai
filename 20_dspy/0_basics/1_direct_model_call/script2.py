import dspy
from dspy import OllamaLocal

# model
model = OllamaLocal(model='mistral:v0.2')
dspy.configure(lm=model)

# ------------------------------------------
# FOR LOOP = call model multiple times 
# ------------------------------------------

# WRONG 1 !!!! 
print("---- Wrong 1 ----")
for idx in range(5):
    response = model("Tell me a joke.")
    print(f'{idx+1}.', response)


# WRONG 3 !!!! 
print("---- Wrong 2 ----")
qa = dspy.Predict('input -> output')

# all answers are the same !!!
for idx in range(5):
    response = qa(input="Tell me a joke.")
    print(f'{idx+1}.', response.output)


# CORRECT 1 !!!! 
print("---- Correct 1 ----")
qa = dspy.Predict('input -> output', n=5) # <------
input = "Tell me a joke."
response = qa(input=input)

print(f"Full response: {response}")
print(f"Input: {input}")
print(f"Output: {response.output}")

for idx, output in enumerate(response.completions):
    print(f'{idx+1}.', output)

