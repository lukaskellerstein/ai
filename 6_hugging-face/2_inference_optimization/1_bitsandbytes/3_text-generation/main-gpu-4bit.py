from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import torch

start = time.time()

# ----------------------------------
# Text generation
# ----------------------------------
# GPU - CUDA
# 4 bit
# ----------------------------------

# Bloomz --

# 1.2GB - works
# model_id = "bigscience/bloomz-560m"
# 2.2GB - works
# model_id = "bigscience/bloomz-1b1"
# 3.5GB - works
# model_id = "bigscience/bloomz-1b7"
# 6.1GB - works
model_id = "bigscience/bloomz-3b"
# 14GB (1x14GB) - does not work (not enough GPU RAM)
# model_id = "bigscience/bloomz-7b1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto", load_in_4bit=True
)

# MosaicML --

# model 5GB = 1x5GB = WORKS
# model_id = "mosaicml/mpt-1b-redpajama-200b"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, trust_remote_code=True, load_in_4bit=True
# )

# model 13GB = 1x10GB + 1x3GB = does not work (not enough GPU RAM)
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
# model_id = "mosaicml/mpt-7b"
# config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
# config.attn_config["attn_impl"] = "triton"
# config.init_device = "cuda:0"  # For fast initialization directly on GPU!
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     config=config,
#     torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
#     trust_remote_code=True,
#     load_in_4bit=True,
# )


# Other --

# model 7GB = 1x7GB = error - GPT is not linear
# model_id = "gpt2-xl"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

# model 9GB = 4x2GB + 1x1GB = WORKS
# model_id = "ethzanalytics/stablelm-tuned-alpha-7b-sharded-8bit"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, load_in_4bit=True
# )

# model 40GB = 8x5GB = error Unrecognized configuration class
# model_id = "google/flan-ul2"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, trust_remote_code=True, load_in_4bit=True
# )

# model 11GB = 1x11GB = does not work
# model_id = "EleutherAI/gpt-neo-2.7B"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, device_map="auto", load_in_4bit=True
# )


# ----------------------------------
inputs = tokenizer.encode(
    "Write a fairy tale about a troll saving a princess from a dangerous dragon. The fairy tale is a masterpiece that has achieved praise worldwide and its moral is 'Heroes Come in All Shapes and Sizes'. Story (in Czech):",
    return_tensors="pt",
).to("cuda")
output = model.generate(inputs, max_new_tokens=500)
print(tokenizer.decode(output[0]))


end = time.time()
print(f"NN takes: {end - start} sec.")
