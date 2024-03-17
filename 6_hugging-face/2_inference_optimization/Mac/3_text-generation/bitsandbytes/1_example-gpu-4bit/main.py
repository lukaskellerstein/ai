import time
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

start = time.time()
import torch
from datetime import timedelta

start = time.time()

# ----------------------------------
# Inference
# ----------------------------------

# Model
model_name = "mistralai/Mistral-7B-v0.1"


# Load the trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=quant_config
)
# ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models. Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
# model_4bit = model_4bit.to("mps")


# ----------------------------------
# Using
# ----------------------------------

# Create a pipeline for text generation
my_pipeline = pipeline("text-generation", model=model_4bit, tokenizer=tokenizer)

result = my_pipeline("In this course, we will teach you how to", max_new_tokens=50)

print(result)

# ----------------------------------
# Total time for the script
td = timedelta(seconds=(time.time() - start))
print(f"Total time: {td}")
