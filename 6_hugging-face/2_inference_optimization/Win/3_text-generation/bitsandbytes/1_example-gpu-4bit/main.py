import time
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
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


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=bnb_config
)


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