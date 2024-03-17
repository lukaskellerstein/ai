import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------
# Load base model
# ---------------------------------------

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, add_bos_token=True, trust_remote_code=True
)

# ---------------------------------------
# Load QLoRA adapter from file
# ---------------------------------------

# Now load the QLoRA adapter from the appropriate checkpoint directory,
# i.e. the best performing model checkpoint:
ft_model = PeftModel.from_pretrained(base_model, "SAVED_TRAINING/checkpoint-500/")


# ---------------------------------------
# Inference
# ---------------------------------------

print("Doc chat inference:")
print(
    "==================================================================================="
)
query = " hi doc, my bmi is 28 what to do?"
eval_prompt = """Patient's Query: {} \n###\n\n""".format(query)
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
# ft_model.eval()
output = ft_model.generate(
    input_ids=model_input["input_ids"].to(device),
    attention_mask=model_input["attention_mask"],
    max_new_tokens=100,
    repetition_penalty=1.17,
)
# with torch.no_grad():

print(tokenizer.decode(output[0], skip_special_tokens=True))


print("Doc chat inference:")
print(
    "==================================================================================="
)
query = " I had a 12 cm lump show up on my inner upper thigh almost overnight and have no idea what it may be.I know my immune system has been very much activated by all the alternative things I have been doing for my advanced prostate cancer and was thinking possibly dead cells and poisens looking for a another way to exit my body.Is this an absurd thought? ###"
eval_prompt = """Patient's Query:\n\n {} ###\n\n""".format(query)
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
# ft_model.eval()
output = ft_model.generate(
    input_ids=model_input["input_ids"].to(device),
    attention_mask=model_input["attention_mask"],
    max_new_tokens=100,
    repetition_penalty=1.15,
)
# with torch.no_grad():

print(tokenizer.decode(output[0], skip_special_tokens=True))
