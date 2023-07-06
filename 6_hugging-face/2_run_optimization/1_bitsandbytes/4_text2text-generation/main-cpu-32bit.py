from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import time

start = time.time()


# ----------------------------------
# Text2Text generation
# ----------------------------------
# CPU
# 32 bit - standard
# ----------------------------------

# Bloomz --------

# 1.2GB - works
# model_id = "bigscience/mt0-small"
# 2.4GB - works
# model_id = "bigscience/mt0-base"
# 5GB - works
# model_id = "bigscience/mt0-large"
# 15GB (2x7.5GB) - does not work (not enough resources ??)
# model_id = "bigscience/mt0-xl"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# inputs = tokenizer.encode(
#     "Write a fairy tale about a troll saving a princess from a dangerous dragon. The fairy tale is a masterpiece that has achieved praise worldwide and its moral is 'Heroes Come in All Shapes and Sizes'. Story (in Czech language)",
#     return_tensors="pt",
# )
# output = model.generate(inputs, max_length=500)
# print(tokenizer.decode(output[0]))

# Google --------

# 300MB - works
# model_id = "google/flan-t5-small"
# 1GB - works
# model_id = "google/flan-t5-base"
# 3.2GB - works
# model_id = "google/flan-t5-large"
# 12GB (1x10GB + 1x2GB) - does not work (not enough resources ??)
model_id = "google/flan-t5-xl"

tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

inputs = tokenizer(
    "translate English to German: How old are you?",
    return_tensors="pt",
).input_ids

output = model.generate(inputs, max_length=500)
print(tokenizer.decode(output[0]))

# ----------------------------------
end = time.time()
print(f"NN takes: {end - start} sec.")
