from vllm import LLM, SamplingParams

# llm = LLM(model="/home/lukas/Models/3_example-qlora/SAVED_FINE-TUNED/MODEL")
llm = LLM(model="lukaskellerstein/mistral-7b-lex-16bit")

sampling_params = SamplingParams(
    max_tokens=100,  # set it same as max_seq_length in SFT Trainer
    temperature=0.1,
    skip_special_tokens=True,
)


# ValueError: Unknown quantization method: bitsandbytes.
# Must be one of ['awq', 'gptq', 'squeezellm', 'marlin']
outputs = llm.generate("In this course, we will teach you how to", sampling_params)

print(outputs)
