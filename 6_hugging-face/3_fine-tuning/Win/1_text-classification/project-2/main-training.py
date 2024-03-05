from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import evaluate
import numpy as np
import time

start = time.time()


# ----------------------------------
# Fine tuning
# ----------------------------------

# ----------------------------------
# https://huggingface.co/docs/transformers/training
# ----------------------------------
model_name = "bert-base-cased"

# ----------------------------------
# Data
# ----------------------------------
dataset = load_dataset("yelp_review_full")


tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


# ----------------------------------
# Model
# ----------------------------------
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)


# ----------------------------------
# Training WITH evaluation (metrics)
# ----------------------------------
metric = evaluate.load("accuracy")


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
