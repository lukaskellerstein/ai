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
import os
import matplotlib.pyplot as plt
from datetime import timedelta

start = time.time()

# ----------------------------------
# Fine tuning
# ----------------------------------

# Model
model_name = "mistralai/Mistral-7B-v0.1"

# ----------------------------------
# Data
# ----------------------------------
dataset = load_dataset("g-ronimo/lfpodcast")
print("----original----")
print(dataset)
# print("----original-sample---")
# print(dataset["train"][:1])

dataset = dataset["train"].train_test_split(test_size=0.1)
print("----split----")
print(dataset)
# print("----split-sample---")
# print(dataset["train"][:1])


def format_conversation(row):
    # Template for conversation turns in ChatML format
    template = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"

    turns = row["conversation"]

    # If Lex is the first speaker, skip his turn to start with Guest's question
    if turns[0]["from"] == "Lex":
        turns = turns[1:]

    conversation = []
    for i in range(0, len(turns), 2):
        # Assuming the conversation always alternates between Guest and Lex
        question = turns[i]  # Guest
        answer = turns[i + 1]  # Lex

        conversation.append(
            template.format(
                q=question["text"],
                a=answer["text"],
            )
        )
    return {"text": "\n".join(conversation)}


dataset = dataset.map(
    format_conversation,
    remove_columns=dataset["train"].column_names,
    num_proc=os.cpu_count(),
)

print("----formatted----")
print(dataset)
# print("----formatted-sample---")
# print(dataset["train"][:1])


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


plot_data_lengths_raw(dataset["train"], dataset["test"])

# ----------------------------------
# Tokenizer
# ----------------------------------

# Load (slow) Tokenizer, fast tokenizer sometimes ignores added tokens
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Add tokens <|im_start|> and <|im_end|>, latter is special eos token
tokenizer.pad_token = "</s>"
tokenizer.add_tokens(["<|im_start|>"])
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))


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


# tokenize
def tokenize(element):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
    )


dataset_tokenized = dataset.map(
    tokenize,
    batched=True,
    num_proc=os.cpu_count(),  # multithreaded
    remove_columns=[
        "text"
    ],  # don't need the strings anymore, we have tokens from here on
)

print("----tokenized----")
print(dataset_tokenized)

# print("----tokenized-sample---")
# print(dataset_tokenized["train"][:1])
# print(len(dataset_tokenized["train"][:1]["input_ids"][0]))


plot_data_lengths(dataset_tokenized["train"], dataset_tokenized["test"])


decoded_list = [tokenizer.decode(x) for x in dataset_tokenized["train"]["input_ids"]]
# print("----decoded-list---")
# print(len(decoded_list))
# print(decoded_list[:1])


def plot_data_decoded_lengths(decoded_list):
    lengths = [len(x) for x in decoded_list]

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color="blue")
    plt.xlabel("Length of decoded text")
    plt.ylabel("Frequency")
    plt.title("Distribution of Lengths of decoded text")
    plt.show()


plot_data_decoded_lengths(decoded_list)


# ----------------------------------
# Model
# ----------------------------------

# Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.resize_token_embeddings(len(tokenizer))
model.config.eos_token_id = tokenizer.eos_token_id
# `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
# model = model.to("cuda")


# ----------------------------------
# Adding the adopter to the layer
# ----------------------------------

# PEFT
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "down_proj",
        "v_proj",
        "gate_proj",
        "o_proj",
        "up_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],  # needed because we added new tokens to tokenizer/model
    task_type="CAUSAL_LM",
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.config.use_cache = False


# ----------------------------------
# Collation
# ----------------------------------
# collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
def collate(elements):
    tokenlist = [e["input_ids"] for e in elements]
    tokens_maxlen = max([len(t) for t in tokenlist])  # length of longest input

    input_ids, labels, attention_masks = [], [], []
    for tokens in tokenlist:
        # how many pad tokens to add for this sample
        pad_len = tokens_maxlen - len(tokens)

        # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content, otherwise 0
        input_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
        labels.append(tokens + [-100] * pad_len)
        attention_masks.append([1] * len(tokens) + [0] * pad_len)

    batch = {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks),
    }
    return batch


# ----------------------------------
# Training WITH evaluation (metrics)
# ----------------------------------

bs = 8  # batch size
ga_steps = 1  # gradient acc. steps
epochs = 5
steps_per_epoch = len(dataset_tokenized["train"]) // (bs * ga_steps)

args = TrainingArguments(
    output_dir="SAVED_TRAINING",
    optim="paged_adamw_32bit",
    num_train_epochs=epochs,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    gradient_accumulation_steps=ga_steps,
    learning_rate=0.0002,
    save_steps=steps_per_epoch,
    evaluation_strategy="steps",
    eval_steps=steps_per_epoch,  # eval and save once per epoch
    logging_steps=10,
    logging_dir="./logs",  # Directory for storing logs
    report_to="tensorboard",
    lr_scheduler_type="constant",
    group_by_length=True,
    fp16=True,
)


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=collate,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
    args=args,
)


# Start TensorBoard before training to monitor it in progress
# %load_ext tensorboard
# %tensorboard --logdir logs


print("Start training...")
startTrain = time.time()
trainer.train()
print(f"Training takes:  {(time.time() - startTrain)} sec.")

# ----------------------------------
# Save the model to the local directory
# ----------------------------------

# This line saves the fine-tuned model in a way that it can be easily re-loaded using the from_pretrained method of the Transformers library. This is useful for deploying the model in different environments or for sharing it with others. The model saved this way includes the architecture along with the learned weights.
trainer.model.save_pretrained("SAVED_PRETRAINED_MODEL")

# This line saves the entire model, including the configuration, tokenizer, and optimizer state, using the Trainer's internal saving mechanism. It's a comprehensive save that's meant for resuming training later or for inference without needing to reconfigure everything. This method might create a slightly larger set of files because it includes additional state information beyond just the model weights and configuration.
trainer.save_model("SAVED_MODEL")

# This line saves the tokenizer used during the training and preprocessing of the data. The tokenizer is crucial for preparing input data for the model in the same way it was done during training. Saving it ensures consistency between training and inference phases, allowing you to encode new data in exactly the same manner as the training data.
tokenizer.save_pretrained("SAVED_TOKENIZER")

# ----------------------------------
# Evaluate
# ----------------------------------
print("Start evaluating...")
startEval = time.time()
print(trainer.evaluate())
print(f"Training takes:  {(time.time() - startEval)} sec.")

# Total time for the script
td = timedelta(seconds=(time.time() - start))
print(f"Total time: {td}")
