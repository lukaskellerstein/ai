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
# Importing the dataset
dataset = load_dataset("mlabonne/guanaco-llama2-1k")
print("----original----")
print(dataset)
print("----original-sample---")
print(dataset["train"][:1])

dataset = dataset["train"].train_test_split(test_size=0.2)
print("----split----")
print(dataset)
print("----split-sample---")
print(dataset["train"][:1])

# ----------------------------------
# Tokenizer
# ----------------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
# tokenizer.add_bos_token, tokenizer.add_eos_token


exit()

# ----------------------------------
# Model
# ----------------------------------

# Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False  # silence the warnings
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
# model = model.to("cuda")


# ----------------------------------
# Adding the adopter to the layer
# ----------------------------------

# PEFT
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


# ----------------------------------
# Training WITH evaluation (metrics)
# ----------------------------------

# Hyperparameters
training_args = TrainingArguments(
    output_dir="SAVED_TRAINING",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
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
