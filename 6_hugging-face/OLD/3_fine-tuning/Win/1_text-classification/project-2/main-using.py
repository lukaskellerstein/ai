from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

id2label = {0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"}
label2id = {"1 star": 0, "2 stars": 1, "3 stars": 2, "4 stars": 3, "5 stars": 4}

tokenizer = AutoTokenizer.from_pretrained("./SAVED_TOKENIZER")
model = AutoModelForSequenceClassification.from_pretrained(
    "./SAVED_MODEL", num_labels=5, id2label=id2label, label2id=label2id
)

my_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
data = [
    # 5 stars
    "dr. goldberg offers everything i look for in a general practitioner. he's nice and easy to talk to without being patronizing; he's always on time in seeing his patients; he's affiliated with a top-notch hospital (nyu) which my parents have explained to me is very important in case something happens and you need surgery; and you can get referrals to see specialists without having to see him first. really, what more do you need? i'm sitting here trying to think of any complaints i have about him, but i'm really drawing a blank.",
    # 1 star
    "I don't know what Dr. Goldberg was like before moving to Arizona, but let me tell you, STAY AWAY from this doctor and this office. I was going to Dr. Johnson before he left and Goldberg took over when Johnson left. He is not a caring doctor. He is only interested in the co-pay and having you come in for medication refills every month. He will not give refills and could less about patients's financial situations. Trying to get your 90 days mail away pharmacy prescriptions through this guy is a joke. And to make matters even worse, his office staff is incompetent. 90% of the time when you call the office, they'll put you through to a voice mail, that NO ONE ever answers or returns your call. Both my adult children and husband have decided to leave this practice after experiencing such frustration. The entire office has an attitude like they are doing you a favor. Give me a break! Stay away from this doc and the practice. You deserve better and they will not be there when you really need them. I have never felt compelled to write a bad review about anyone until I met this pathetic excuse for a doctor who is all about the money.",
]

result = my_pipeline(data)

print(result)
