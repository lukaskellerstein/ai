import time
from transformers import pipeline


start = time.time()


# ----------------------------------
# Inference
# ----------------------------------


my_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")
result = my_pipeline("In this course, we will teach you how to", max_new_tokens=50)

print(result)


# ----------------------------------
end = time.time()
print(f"NN takes: {end - start} sec.")
