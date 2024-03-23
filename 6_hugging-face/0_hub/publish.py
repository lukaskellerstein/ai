# from huggingface_hub import HfApi
from dotenv import load_dotenv, find_dotenv
import os
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch

_ = load_dotenv(find_dotenv())  # read local .env file

API_TOKEN = os.environ.get("HF_TOKEN")
print(API_TOKEN)

# Model
model_path = "/home/lukas/Models/3_example-qlora/SAVED_FINE-TUNED/ALL"


# Load the trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", quantization_config=bnb_config
)


model_4bit.push_to_hub(
    repo_id="lukaskellerstein/mistral-7b-lex-4bit",
    token=API_TOKEN,
    commit_message="Initial commit",
)
tokenizer.push_to_hub(
    repo_id="lukaskellerstein/mistral-7b-lex-4bit",
    token=API_TOKEN,
)
