from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, load_dataset, DatasetDict
import numpy as np
import time
import evaluate
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os
import matplotlib.pyplot as plt

start = time.time()

# ----------------------------------
# Fine tuning
# ----------------------------------

# Model
model_name = "mistralai/Mistral-7B-v0.1"

# ----------------------------------
# Data
# ----------------------------------
dataset = load_dataset("json", data_files="data/all.json")
dataset = dataset["train"].train_test_split(test_size=0.3)
print(dataset)


# ----------------------------
# Plot distribution of length of text
# ----------------------------
def plot_data_lengths(dataset: DatasetDict, fieldName: str):
    lengths = [len(x[fieldName]) for x in dataset["train"]]
    lengths += [len(x[fieldName]) for x in dataset["test"]]

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color="blue")
    plt.xlabel("Length of Text")
    plt.ylabel("Frequency")
    plt.title("Distribution of Lengths of Text")
    plt.show()


plot_data_lengths(dataset, "text")


# ----------------------------------
# Tokenizer
# ----------------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(element):
    return tokenizer(
        element["text"],
        # truncation=True,
        # max_length=2048,
        # add_special_tokens=False,
    )


dataset_tokenized = dataset.map(
    tokenize,
    batched=True,
    num_proc=os.cpu_count(),  # multithreaded
    remove_columns=[
        "text"
    ],  # don't need the strings anymore, we have tokens from here on
)

print(dataset_tokenized)


# ----------------------------
# Plot distribution of length of input_ids
# ----------------------------
def plot_data_lengths_tokenized(dataset: DatasetDict):
    lengths = [len(x["input_ids"]) for x in dataset["train"]]
    lengths += [len(x["input_ids"]) for x in dataset["test"]]

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color="blue")
    plt.xlabel("Length of Input_ids")
    plt.ylabel("Frequency")
    plt.title("Distribution of Lengths Input_ids")
    plt.show()


plot_data_lengths_tokenized(dataset_tokenized)


exit()

# # ----------------------------------
# # Model
# # ----------------------------------

# # Quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )


# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True,
# )
# model.config.use_cache = False  # silence the warnings
# model.config.pretraining_tp = 1
# model.gradient_checkpointing_enable()

# # `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
# # model = model.to("cuda")


# # ----------------------------------
# # Collation
# # ----------------------------------
# # # collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
# # def collate(elements):
# #     tokenlist = [e["input_ids"] for e in elements]
# #     tokens_maxlen = max([len(t) for t in tokenlist])  # length of longest input

# #     input_ids, labels, attention_masks = [], [], []
# #     for tokens in tokenlist:
# #         # how many pad tokens to add for this sample
# #         pad_len = tokens_maxlen - len(tokens)

# #         # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content, otherwise 0
# #         input_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
# #         labels.append(tokens + [-100] * pad_len)
# #         attention_masks.append([1] * len(tokens) + [0] * pad_len)

# #     batch = {
# #         "input_ids": torch.tensor(input_ids),
# #         "labels": torch.tensor(labels),
# #         "attention_mask": torch.tensor(attention_masks),
# #     }
# #     return batch


# # ----------------------------------
# # Adding the adopter to the layer
# # ----------------------------------

# # PEFT
# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
# )

# # Prepare model for k-bit training
# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, peft_config)


# # ----------------------------------
# # Training WITH evaluation (metrics)
# # ----------------------------------

# bs = 8  # batch size
# ga_steps = 1  # gradient acc. steps
# epochs = 2  # number of epochs
# steps_per_epoch = len(dataset_tokenized["train"]) / (bs * ga_steps)

# training_args = TrainingArguments(
#     output_dir="SAVED_TRAINING",
#     per_device_train_batch_size=bs,
#     per_device_eval_batch_size=bs,
#     evaluation_strategy="steps",
#     logging_steps=1,
#     eval_steps=steps_per_epoch,  # eval and save once per epoch
#     save_steps=steps_per_epoch,
#     gradient_accumulation_steps=ga_steps,
#     num_train_epochs=epochs,
#     lr_scheduler_type="constant",
#     optim="paged_adamw_32bit",
#     learning_rate=0.0002,
#     group_by_length=True,
#     fp16=True,
#     ddp_find_unused_parameters=False,  # needed for training with accelerate
# )

# trainer = Trainer(
#     model=model,
#     tokenizer=tokenizer,
#     data_collator=collate,
#     train_dataset=dataset_tokenized["train"],
#     eval_dataset=dataset_tokenized["test"],
#     args=training_args,
# )


# print("Start training...")
# startTrain = time.time()
# trainer.train()
# print(f"Training takes:  {(time.time() - startTrain)} sec.")

# # ----------------------------------
# # Save the model to the local directory
# # ----------------------------------

# trainer.save_model("SAVED_MODEL")
# tokenizer.save_pretrained("SAVED_TOKENIZER")

# # ----------------------------------
# # Evaluate
# # ----------------------------------
# print("Start evaluating...")
# startEval = time.time()
# print(trainer.evaluate())
# print(f"Training takes:  {(time.time() - startEval)} sec.")

# # Total time for the script
# print(f"Total time:  {(time.time() - start)} sec.")
