import time
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
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

# Define the prompt
prompt = "In this course, we will teach you how to"

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output = model_4bit.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode and print the generated text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)

# ----------------------------------
# Total time for the script
td = timedelta(seconds=(time.time() - start))
print(f"Total time: {td}")
