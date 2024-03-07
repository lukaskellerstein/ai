import time
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

start = time.time()


# ----------------------------------
# Inference
# ----------------------------------

# Model
model_name = "mistralai/Mistral-7B-v0.1"


# Load the trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quant_config)
# model_4bit = model_4bit.to("mps")

# Create a pipeline for text generation
my_pipeline = pipeline("text-generation", 
                       model=model_4bit, 
                       tokenizer=tokenizer)

result = my_pipeline("In this course, we will teach you how to", max_new_tokens=50)

print(result)

# ----------------------------------
end = time.time()
print(f"NN takes: {end - start} sec.")
