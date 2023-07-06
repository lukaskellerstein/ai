from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

tokenizer = AutoTokenizer.from_pretrained("./SAVED_TOKENIZER")
model = AutoModelForSequenceClassification.from_pretrained(
    "./SAVED_MODEL", num_labels=2, id2label=id2label, label2id=label2id
)

my_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
data = [
    "This is such amazing movie!",
    "Horrible movie!",
    "Bad",
]

result = my_pipeline(data)
print(result)

# negative
print(
    my_pipeline(
        "No magic here, it was all I could do to keep from turning it off an hour in."
    )
)
# positive
print(
    my_pipeline(
        "This movie was great, and I suggest that you go see it before you judge."
    )
)
