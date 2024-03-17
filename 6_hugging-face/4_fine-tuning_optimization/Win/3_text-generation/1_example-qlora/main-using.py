import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
from datetime import timedelta

start = time.time()


# ----------------------------------
# Data
# ----------------------------------

# Load the trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("SAVED_TOKENIZER")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ----------------------------------
# Model
# ----------------------------------

# Quantization
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "SAVED_MODEL", device_map="auto", quantization_config=nf4_config
)

# `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
# model = model.to("cuda")


# ----------------------------------
# Inference
# ----------------------------------

# Create a pipeline for text generation
my_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Use the model to generate responses
input_text = "### Instruction:\nUse the provided input to create an instruction that could have been used to generate the response with an LLM.### Input:\nThere are more than 12,000 species of grass. The most common is Kentucky Bluegrass, because it grows quickly, easily, and is soft to the touch. Rygrass is shiny and bright green colored. Fescues are dark green and shiny. Bermuda grass is harder but can grow in drier soil.\n\n### Response:"
generated_text = my_pipeline(input_text, max_new_tokens=100)[0]["generated_text"]

print(generated_text)


td = timedelta(seconds=(time.time() - start))
print(f"NN takes: {td}")
