from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# T5-3b and T5-11B are supported!
# We need sharded weights otherwise we get CPU OOM errors
model_id = "ybelkada/t5-3b-sharded"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model_8bit = AutoModelForSeq2SeqLM.from_pretrained(
    model_id, device_map="auto", load_in_8bit=True
)


print(model_8bit.get_memory_footprint())

max_new_tokens = 50

input_ids = tokenizer(
    "translate English to German: Hello my name is Younes and I am a Machine Learning Engineer at Hugging Face",
    return_tensors="pt",
).input_ids

outputs = model_8bit.generate(input_ids, max_new_tokens=max_new_tokens)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
