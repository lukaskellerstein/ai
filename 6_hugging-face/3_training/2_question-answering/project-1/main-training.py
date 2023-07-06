import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import notebook_login
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
import os
import time

_ = load_dotenv(find_dotenv())  # read local .env file


start = time.time()


# ----------------------------------
# Fine tuning
# ----------------------------------

# ----------------------------------
# https://www.youtube.com/watch?v=jPYs0ecJyQ0
# ----------------------------------
model_name = "Helsinki-NLP/opus-mt-en-es"

# ----------------------------------
# Data
# ----------------------------------

data = load_dataset("kde4", lang1="en", lang2="es")

small = data["train"].shuffle(seed=42).select(range(1_000))
split = small.train_test_split(seed=42, test_size=0.2)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

en_sample = split["train"][5]["translation"]["en"]
es_sample = split["train"][5]["translation"]["es"]

inputs = tokenizer(en_sample)
targets = tokenizer(text_target=es_sample)

# ----------------------------------
# Analyze Token length
# ----------------------------------

train = split["train"]["translation"]
input_lens = [len(tr["en"]) for tr in train]
output_lens = [len(tr["es"]) for tr in train]

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist(input_lens, bins=50, label="input", color="orange")
plt.title("Input Lengths")

plt.subplot(1, 2, 2)
plt.hist(output_lens, bins=50, label="output")
plt.title("Output Lengths")

plt.show()

input_lens = pd.Series(input_lens)
output_lens = pd.Series(output_lens)

print("## Input Lengths ##")
print(input_lens.describe())

print("\n## Output Lengths ##")
print(output_lens.describe())


max_input_len = 128
max_target_len = 128


def tokenizer_fn(batch):
    inputs = [x["en"] for x in batch["translation"]]
    targets = [x["es"] for x in batch["translation"]]

    tokenized_inputs = tokenizer(inputs, max_length=max_input_len, truncation=True)

    tokenized_targets = tokenizer(
        text_target=targets, max_length=max_target_len, truncation=True
    )

    tokenizer_full = tokenized_inputs.copy()
    tokenizer_full["labels"] = tokenized_targets["input_ids"]
    return tokenizer_full


tokenized_datasets = split.map(
    tokenizer_fn,
    batched=True,
    remove_columns=split["train"].column_names,
)


# ----------------------------------
# Evaluation metric
# ----------------------------------

# just compare 2 metrics

sentence1 = "I'm so hungry"
sentence2 = ["I'm starving"]

bleu_metric = load_metric("sacrebleu")
bert_metric = load_metric("bertscore")

print(
    "BLEU Score : ",
    bleu_metric.compute(predictions=[sentence1], references=[sentence2]),
)
print(
    "BERT Score : ",
    bert_metric.compute(predictions=[sentence1], references=[sentence2], lang="en"),
)


def compute_metrics(
    preds_and_labels,
    bleu_metric=load_metric("sacrebleu"),
    bert_metric=load_metric("bertscore"),
):
    preds, labels = preds_and_labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # for any -100 label, replace with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # convert labels into words
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    bert_score = bert_metric.compute(
        predictions=decoded_preds, references=decoded_labels, lang="fr"
    )

    return {"bleu": bleu["score"], "bert_score": np.mean(bert_score["f1"])}


# ----------------------------------
# Train HF NMT Model
# ----------------------------------

Seq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=Seq2SeqLM)

training_args = Seq2SeqTrainingArguments(
    output_dir="SAVED_TRAINING",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=Seq2SeqLM,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()

print(trainer.evaluate(max_length=max_target_len))


# ----------------------------------
# Save the model to the local directory
# ----------------------------------
trainer.save_model("SAVED_MODEL")
tokenizer.save_pretrained("SAVED_TOKENIZER")

# ----------------------------------
# Publish the model to the Hugging Face Hub
# ----------------------------------

# token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
# print(token)
# login(token=token)


# trainer.push_to_hub("my-first-model")


end = time.time()
print(f"NN takes: {end - start} sec.")
