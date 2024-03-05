from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


tokenizer = AutoTokenizer.from_pretrained("./SAVED_TOKENIZER")
model = AutoModelForSeq2SeqLM.from_pretrained("./SAVED_MODEL")

translator = pipeline("translation", model=model, tokenizer=tokenizer, device=-1)
result = translator("Hell yeah ! I finetuned Hugging Face NMT model !", max_length=128)

print(result)
