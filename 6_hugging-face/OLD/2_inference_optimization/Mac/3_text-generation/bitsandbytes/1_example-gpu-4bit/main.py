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


# ------------------------------------------------------------
# ------------------------------------------------------------
# DOES NOT WORK ON MAC !!!!!
# ERROR: 
# ImportError: Using `bitsandbytes` 8-bit quantization requires Accelerate: `pip install accelerate` and the latest version of bitsandbytes: `pip install -i https://pypi.org/simple/ bitsandbytes`
# ------------------------------------------------------------
# ------------------------------------------------------------



# ----------------------------------
# Inference
# ----------------------------------

# Model
model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"


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
# ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models. Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
# model_4bit = model_4bit.to("mps")




# ----------------------------------
# Using Model and Tokenizer directly
# ----------------------------------

# # Define the prompt
# prompt = "In this course, we will teach you how to"

# # Tokenize the prompt
# input_ids = tokenizer.encode(prompt, return_tensors="pt")

# # Generate text
# output = model_4bit.generate(input_ids, max_length=100, num_return_sequences=1)

# # Decode and print the generated text
# output_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(output_text)

# ----------------------------------
# Using Pipeline
# ----------------------------------

# # Create a pipeline for text generation
# my_pipeline = pipeline("text-generation", model=model_4bit, tokenizer=tokenizer)

# result = my_pipeline("In this course, we will teach you how to", max_new_tokens=50)

# print(result)




# ----------------------------------
# CHAT - Using Model and Tokenizer directly
# ----------------------------------


# def encode_chat(chat):
#     # Apply the chat template
#     formatted_chat = tokenizer.apply_chat_template(
#         chat, tokenize=False, add_generation_prompt=False
#     )

#     # Tokenize the formatted chat
#     # This should return a dictionary with keys like 'input_ids', 'attention_mask', etc.
#     encoded_chat = tokenizer(
#         formatted_chat, truncation=True, padding="max_length", max_length=512
#     )

#     return encoded_chat


# default chat template format
print("----- Default chat template -----")
print(tokenizer.default_chat_template)

chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

result = tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=False
)
print("----- Applied on data -----")
print(result)


print("----- Encoded -----")
encoded_chat = tokenizer(
        result, truncation=True, padding="max_length", max_length=512
    )
print(encoded_chat)



print("----- GENERATE -----")
# Generate text
output = model_4bit.generate(encoded_chat, max_length=100, num_return_sequences=1)

print("----- Decode -----")
# Decode and print the generated text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)



# ----------------------------------
# Total time for the script
td = timedelta(seconds=(time.time() - start))
print(f"Total time: {td}")
