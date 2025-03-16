from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import time
import evaluate
from data import train_data, eval_data
import torch

start = time.time()

# ----------------------------------
# Fine tuning
# ----------------------------------

# Model
model_name = "mistralai/Mistral-7B-v0.1"

# ----------------------------------
# Data
# ----------------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def encode_chat(chat):
    # Apply the chat template
    formatted_chat = tokenizer.apply_chat_template(
        chat["chat"], tokenize=False, add_generation_prompt=False
    )

    # Tokenize the formatted chat
    # This should return a dictionary with keys like 'input_ids', 'attention_mask', etc.
    encoded_chat = tokenizer(
        formatted_chat, truncation=True, padding="max_length", max_length=512
    )

    # Add labels for language modeling
    encoded_chat["labels"] = torch.tensor(encoded_chat["input_ids"]).clone()

    return encoded_chat


# Train dataset
train_dataset = Dataset.from_dict({"chat": [train_data]})
train_dataset = train_dataset.map(encode_chat)

# Eval dataset
eval_dataset = Dataset.from_dict({"chat": [eval_data]})
eval_dataset = eval_dataset.map(encode_chat)

# ----------------------------------
# Model
# ----------------------------------
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to("mps")

# Model architecture
print(model.config)


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
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

print(trainer.evaluate())


# ----------------------------------
# Save the model to the local directory
# ----------------------------------

trainer.save_model("SAVED_MODEL")
tokenizer.save_pretrained("SAVED_TOKENIZER")

end = time.time()
print(f"NN takes: {end - start} sec.")

# {'eval_runtime': 2.5533, 'eval_samples_per_second': 0.392, 'eval_steps_per_second': 0.392}
