from transformers import pipeline

generator = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")
result = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

print(result)
