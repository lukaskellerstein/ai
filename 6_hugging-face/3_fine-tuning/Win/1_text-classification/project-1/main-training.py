from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import time


start = time.time()


# ----------------------------------
# Fine tuning
# ----------------------------------

# ----------------------------------
# https://huggingface.co/transformers/v4.8.2/training.html
# ----------------------------------
model_name = "bert-base-cased"

# ----------------------------------
# Data
# ----------------------------------
raw_datasets = load_dataset("imdb")

print(raw_datasets)

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]


# ----------------------------------
# Model
# ----------------------------------

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# ----------------------------------
# Training without evaluation (metrics)
# ----------------------------------
# training_args = TrainingArguments("test_trainer")

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
# )

# trainer.train()


# ----------------------------------
# Training WITH evaluation (metrics)
# ----------------------------------
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="SAVED_TRAINING", evaluation_strategy="epoch"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

print(trainer.evaluate())


# ----------------------------------
# Save the model to the local directory
# ----------------------------------

trainer.save_model("SAVED_MODEL")
tokenizer.save_pretrained("SAVED_TOKENIZER")

end = time.time()
print(f"NN takes: {end - start} sec.")
