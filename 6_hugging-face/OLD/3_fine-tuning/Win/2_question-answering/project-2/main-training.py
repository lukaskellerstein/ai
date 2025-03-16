import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import notebook_login
from datasets import load_dataset, load_metric
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    AdamW,
)
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
import os
import time
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

_ = load_dotenv(find_dotenv())  # read local .env file


start = time.time()


# ----------------------------------
# Fine tuning with custom dataset
# ----------------------------------

# ----------------------------------
# https://huggingface.co/transformers/v3.1.0/custom_datasets.html#question-answering-with-squad-2-0
# ----------------------------------
model_name = "distilbert-base-uncased"


# ----------------------------------
# Data
# ----------------------------------
def read_squad(path):
    path = Path(path)
    with open(path, "rb") as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict["data"]:
        for passage in group["paragraphs"]:
            context = passage["context"]
            for qa in passage["qas"]:
                question = qa["question"]
                for answer in qa["answers"]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


train_contexts, train_questions, train_answers = read_squad("data/train-v2.0.json")
val_contexts, val_questions, val_answers = read_squad("data/dev-v2.0.json")


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer["text"]
        start_idx = answer["answer_start"]
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer["answer_end"] = end_idx
        elif context[start_idx - 1 : end_idx - 1] == gold_text:
            answer["answer_start"] = start_idx - 1
            answer["answer_end"] = (
                end_idx - 1
            )  # When the gold label is off by one character
        elif context[start_idx - 2 : end_idx - 2] == gold_text:
            answer["answer_start"] = start_idx - 2
            answer["answer_end"] = (
                end_idx - 2
            )  # When the gold label is off by two characters


add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

# tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

train_encodings = tokenizer(
    train_contexts, train_questions, truncation=True, padding=True
)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]["answer_start"]))
        end_positions.append(encodings.char_to_token(i, answers[i]["answer_end"] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update(
        {"start_positions": start_positions, "end_positions": end_positions}
    )


add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)


# ----------------------------------
# Model
# ----------------------------------
model = DistilBertForQuestionAnswering.from_pretrained(model_name)


# ----------------------------------
# Training (manual Pytorch)
# ----------------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        loss = outputs[0]
        loss.backward()
        optim.step()

print(model.eval())


# ----------------------------------
# Save the model to the local directory
# ----------------------------------

model.save_pretrained("SAVED_MODEL")
tokenizer.save_pretrained("SAVED_TOKENIZER")


end = time.time()
print(f"NN takes: {end - start} sec.")
