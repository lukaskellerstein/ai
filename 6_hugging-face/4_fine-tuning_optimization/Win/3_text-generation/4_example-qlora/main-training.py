from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, load_dataset
import numpy as np
import time
import evaluate
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import matplotlib.pyplot as plt
import json

start = time.time()


def logData(name, dataset, count=10):
    result = []

    for i in range(1, count):
        result.append(dataset[i])

    with open(f"log/{name}.json", "w") as f:
        json.dump(result, f)


# ----------------------------------
# Fine tuning
# ----------------------------------

# Model
model_name = "mistralai/Mistral-7B-v0.1"

# ----------------------------------
# Data
# ----------------------------------
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
print("----original----")
print(dataset)
logData("data-origin", dataset["train"])

dataset = dataset["train"].train_test_split(test_size=0.3)
train_dataset = dataset["train"].select(range(2000))
eval_dataset = dataset["test"].select(range(500))


def formatting_func(example):
    text = f"### The following is a doctor's opinion on a person's query: \n### Patient query: {example['input']} \n### Doctor opinion: {example['output']}"
    return {"text": text}


train_dataset = train_dataset.map(
    formatting_func,
    remove_columns=dataset["train"].column_names,
    num_proc=os.cpu_count(),
)
eval_dataset = eval_dataset.map(
    formatting_func,
    remove_columns=dataset["test"].column_names,
    num_proc=os.cpu_count(),
)

print("----formatted----")
print(train_dataset)
print(eval_dataset)
logData("formatted", train_dataset)


def plot_data_lengths_raw(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x["text"]) for x in tokenized_train_dataset]
    lengths += [len(x["text"]) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color="blue")
    plt.xlabel("Length of text")
    plt.ylabel("Frequency")
    plt.title("Distribution of Lengths of text")
    plt.show()


# plot_data_lengths_raw(train_dataset, eval_dataset)


# ----------------------------------
# Tokenizer
# ----------------------------------

# Load (slow) Tokenizer, fast tokenizer sometimes ignores added tokens
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token


def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x["input_ids"]) for x in tokenized_train_dataset]
    lengths += [len(x["input_ids"]) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color="blue")
    plt.xlabel("Length of input_ids")
    plt.ylabel("Frequency")
    plt.title("Distribution of Lengths of input_ids")
    plt.show()


# ----------------------------
# Tokenizing
# ----------------------------

max_length = 512  # This was an appropriate max length for my dataset


def generate_and_tokenize_prompt2(prompt):
    # print("prompt")
    # print(prompt)
    result = tokenizer(
        prompt["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_train_dataset = train_dataset.map(
    generate_and_tokenize_prompt2,
    remove_columns=[
        "text"
    ],  # don't need the strings anymore, we have tokens from here on
)
tokenized_val_dataset = eval_dataset.map(
    generate_and_tokenize_prompt2,
    remove_columns=[
        "text"
    ],  # don't need the strings anymore, we have tokens from here on)
)

print("----tokenized---")
print(tokenized_train_dataset)
print(tokenized_val_dataset)
logData("tokenized", tokenized_train_dataset)


decoded_text = tokenizer.decode(tokenized_train_dataset[:1]["input_ids"][0])
print("----decoded-text---")
print(decoded_text)

decoded_list = [tokenizer.decode(x) for x in tokenized_train_dataset[:10]["input_ids"]]
logData("decoded", decoded_list)

# plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)


# exit()

# ----------------------------------
# Model
# ----------------------------------

# Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
# model = model.to("cuda")


# ----------------------------------
# Adding the adopter to the layer
# ----------------------------------

# PEFT
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

# Prepare model for k-bit training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print_trainable_parameters(model)

# ----------------------------------
# Training WITH evaluation (metrics)
# ----------------------------------
epochs = 2
bs = 3  # batch size
ga_steps = 1  # gradient acc. steps

args = TrainingArguments(
    output_dir="SAVED_TRAINING",
    optim="paged_adamw_8bit",
    num_train_epochs=epochs,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    gradient_accumulation_steps=ga_steps,
    learning_rate=2.5e-4,  # Want a small lr for finetuning
    save_strategy="steps",  # Save the model checkpoint every logging step
    save_steps=25,  # Save checkpoints every 50 steps
    evaluation_strategy="steps",  # Evaluate the model every logging step
    eval_steps=25,  # Evaluate and save checkpoints every 50 steps
    logging_steps=25,  # When to start reporting loss
    logging_dir="./logs",  # Directory for storing logs
    warmup_steps=1,
    gradient_checkpointing=True,
    max_steps=500,
    # bf16=True,
    do_eval=True,  # Perform evaluation at the end of training
)


trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


print("Start training...")
startTrain = time.time()
trainer.train()
print(f"Training takes:  {(time.time() - startTrain)} sec.")

# ----------------------------------
# Save the model to the local directory
# ----------------------------------
trainer.model.save_pretrained("SAVED_PRETRAINED_MODEL")
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
