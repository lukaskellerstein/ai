from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
import numpy as np
import time
import evaluate


start = time.time()


# ----------------------------------
# Fine tuning with custom dataset
# ----------------------------------

# ----------------------------------
# https://huggingface.co/transformers/v3.1.0/custom_datasets.html
# ----------------------------------
model_name = "distilbert-base-uncased"


# ----------------------------------
# Data
# ----------------------------------

# http://ai.stanford.edu/~amaas/data/sentiment/


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir / label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels


train_texts, train_labels = read_imdb_split("data/aclImdb/train")
test_texts, test_labels = read_imdb_split("data/aclImdb/test")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2
)


class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

# ----------------------------------
# Model
# ----------------------------------
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# ----------------------------------
# Training without evaluation (metrics)
# ----------------------------------
# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=2,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=64,
#     warmup_steps=500,
#     learning_rate=5e-5,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=10,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# trainer.train()


# ----------------------------------
# Training WITH evaluation (metrics)
# ----------------------------------
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="SAVED_TRAINING",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
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
