import torch
import transformers

name = "mosaicml/mpt-7b"

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config["attn_impl"] = "triton"
config.init_device = "cuda:0"  # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    trust_remote_code=True,
)

tokenizer = transformers.AutoTokenizer.from_pretrained(name, trust_remote_code=True)

# Generate text
input_ids = tokenizer("The dog", return_tensors="pt").input_ids.to("cuda:0")
output = model.generate(
    input_ids, do_sample=True, max_length=50, num_return_sequences=1
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
