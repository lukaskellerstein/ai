from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, load_dataset
import numpy as np
import time
import evaluate
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

start = time.time()

# ----------------------------------
# Fine tuning
# ----------------------------------

# Model
model_name = "mistralai/Mistral-7B-v0.1"

# ----------------------------------
# Data
# ----------------------------------
instruct_tune_dataset = load_dataset("mosaicml/instruct-v3")
instruct_tune_dataset = instruct_tune_dataset.filter(
    lambda x: x["source"] == "dolly_hhrlhf"
)
instruct_tune_dataset["train"] = instruct_tune_dataset["train"].select(range(3000))
instruct_tune_dataset["test"] = instruct_tune_dataset["test"].select(range(200))


def create_prompt(sample):
    bos_token = "<s>"
    original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    system_message = "Use the provided input to create an instruction that could have been used to generate the response with an LLM."
    response = (
        sample["prompt"]
        .replace(original_system_message, "")
        .replace("\n\n### Instruction\n", "")
        .replace("\n### Response\n", "")
        .strip()
    )
    input = sample["response"]
    eos_token = "</s>"

    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "### Instruction:"
    full_prompt += "\n" + system_message
    full_prompt += "\n\n### Input:"
    full_prompt += "\n" + input
    full_prompt += "\n\n### Response:"
    full_prompt += "\n" + response
    full_prompt += eos_token

    return full_prompt


tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    model_name, device_map="auto", quantization_config=nf4_config, use_cache=False
)

# `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
# model = model.to("cuda")

# ----------------------------------
# Training WITH evaluation (metrics)
# ----------------------------------

# PEFT
peft_config = LoraConfig(
    lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM"
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


training_args = TrainingArguments(
    output_dir="SAVED_TRAINING",
    # num_train_epochs=5,
    max_steps=2,  # originally 100
    per_device_train_batch_size=4,
    warmup_steps=0.03,
    logging_steps=10,
    save_strategy="epoch",
    # evaluation_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=20,
    learning_rate=2e-4,
    bf16=True,
    lr_scheduler_type="constant",
)

max_seq_length = 2048

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=create_prompt,
    args=training_args,
    train_dataset=instruct_tune_dataset["train"],
    eval_dataset=instruct_tune_dataset["test"],
)

print("Start training...")
startTrain = time.time()
trainer.train()
print(f"Training takes:  {(time.time() - startTrain)} sec.")

# ----------------------------------
# Save the model to the local directory
# ----------------------------------

trainer.model.save_pretrained("SAVED_MODEL_ONLY")

trainer.save_model("SAVED_MODEL")
tokenizer.save_pretrained("SAVED_TOKENIZER")

# ----------------------------------
# Evaluate
# ----------------------------------
print("Start evaluating...")
startEval = time.time()
print(trainer.evaluate())
print(f"Training takes:  {(time.time() - startEval)} sec.")

# Total time for the script
print(f"Total time:  {(time.time() - start)} sec.")
