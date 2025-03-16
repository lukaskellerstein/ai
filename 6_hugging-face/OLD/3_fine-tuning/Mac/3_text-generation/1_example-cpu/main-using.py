import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


start = time.time()

# Load the trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("SAVED_MODEL")
tokenizer = AutoTokenizer.from_pretrained("SAVED_TOKENIZER")

# Create a pipeline for text generation
my_pipeline = pipeline("text-generation", 
                       model=model, 
                       tokenizer=tokenizer)

# Your new sentences
data = [
    {"role": "user", "content": "Which is bigger, the moon or the earth?"},
    {"role": "user", "content": "Which is bigger, a atom or a virus?"}
]

# Use the model to generate responses
for chat in data:
    input_text = chat["content"]
    generated_text = my_pipeline(input_text, pad_token_id=my_pipeline.tokenizer.eos_token_id, max_new_tokens=50)[0]['generated_text']
    print(f"User: {input_text}\nAssistant: {generated_text}\n")


end = time.time()
print(f"NN takes: {end - start} sec.")
