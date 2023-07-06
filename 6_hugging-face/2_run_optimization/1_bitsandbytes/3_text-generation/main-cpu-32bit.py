from transformers import AutoModelForCausalLM, AutoTokenizer
import time

start = time.time()

# ----------------------------------
# Text generation
# ----------------------------------
# CPU
# 32 bit - standard
# ----------------------------------

# Bloomz --

# 1.2GB - works
# model_id = "bigscience/bloomz-560m"
# 2.2GB
# model_id = "bigscience/bloomz-1b1"
# 3.5GB - does not work (not enough RAM)
model_id = "bigscience/bloomz-1b7"
# 6.1GB - does not work (not enough RAM)
# model_id = "bigscience/bloomz-3b"
# 14GB (1x14GB) - does not work (not enough RAM)
# model_id = "bigscience/bloomz-7b1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# ----------------------------------
inputs = tokenizer.encode(
    "Write a fairy tale about a troll saving a princess from a dangerous dragon. The fairy tale is a masterpiece that has achieved praise worldwide and its moral is 'Heroes Come in All Shapes and Sizes'. Story (in Czech):",
    return_tensors="pt",
)
output = model.generate(inputs, max_new_tokens=500)
print(tokenizer.decode(output[0]))


end = time.time()
print(f"NN takes: {end - start} sec.")
