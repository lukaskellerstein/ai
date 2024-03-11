from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import numpy as np
import time
import evaluate

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

    return encoded_chat


# Train dataset
chat1 = [
    {"role": "user", "content": "Which is bigger, the moon or the sun?"},
    {"role": "assistant", "content": "The sun."},
]
chat2 = [
    {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
    {"role": "assistant", "content": "A bacterium."},
]

train_dataset = Dataset.from_dict({"chat": [chat1, chat2]})
train_dataset = train_dataset.map(encode_chat)

# Eval dataset
chat3 = [
    {"role": "user", "content": "Which is hotter, the sun or the earth?"},
    {"role": "assistant", "content": "The sun."},
]
chat4 = [
    {"role": "user", "content": "Which is bigger, an atom or a molecule?"},
    {"role": "assistant", "content": "A molecule."},
]

eval_dataset = Dataset.from_dict({"chat": [chat3, chat4]})
eval_dataset = eval_dataset.map(encode_chat)


# ----------------------------------
# Model
# ----------------------------------
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to("cuda")

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

print(trainer.evaluate())


# ----------------------------------
# Save the model to the local directory
# ----------------------------------

trainer.save_model("SAVED_MODEL")
tokenizer.save_pretrained("SAVED_TOKENIZER")

end = time.time()
print(f"NN takes: {end - start} sec.")
