import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


start = time.time()

# Load the trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("SAVED_MODEL")
model = model.to("mps")

tokenizer = AutoTokenizer.from_pretrained("SAVED_TOKENIZER")

# Create a pipeline for text generation
my_pipeline = pipeline("text-generation", 
                       model=model, 
                       tokenizer=tokenizer, 
                       device="mps")

# Your new sentences
data = [
    {"role": "user", "content": "What's your take on historical documents?"},
    {"role": "user", "content": "How do you view the role of technology in our lives?"},
]

# Use the model to generate responses
for chat in data:
    input_text = chat["content"]
    generated_text = my_pipeline(input_text, pad_token_id=my_pipeline.tokenizer.eos_token_id)[0]['generated_text']
    print(f"User: {input_text}\nAssistant: {generated_text}\n")


end = time.time()
print(f"NN takes: {end - start} sec.")
