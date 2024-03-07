import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


start = time.time()


# ----------------------------------
# Inference
# ----------------------------------

# Model
model_name = "mistralai/Mistral-7B-v0.1"


# Load the trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to("mps")

# Create a pipeline for text generation
my_pipeline = pipeline("text-generation", 
                       model=model, 
                       tokenizer=tokenizer,
                       device="mps")

result = my_pipeline("In this course, we will teach you how to", max_new_tokens=50)

print(result)

# ----------------------------------
end = time.time()
print(f"NN takes: {end - start} sec.")
