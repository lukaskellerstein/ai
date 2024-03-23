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
model_path = "/home/lukas/Models/3_example-qlora/SAVED_FINE-TUNED/MODEL"


# Load the trained model and tokenizer
tokenizer_path = "/home/lukas/Models/3_example-qlora/SAVED_FINE-TUNED/TOKENIZER"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", quantization_config=bnb_config
)


# ----------------------------------
# Using Model and Tokenizer directly
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
# Using Pipeline
# ----------------------------------
prompt = "In this course, we will teach you how to"

# Create a pipeline for text generation
my_pipeline = pipeline("text-generation", model=model_4bit, tokenizer=tokenizer)

output_text = my_pipeline(prompt, max_new_tokens=100)
print(output_text)

# ----------------------------------
# Total time for the script
td = timedelta(seconds=(time.time() - start))
print(f"Total time: {td}")
