# Inference Locally - Ollama:

Works only 4bit model , other need more resources

1. Download model from HuggingFace:
   GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:lukaskellerstein/mistral-7b-lex-32bit
   Then download files manually from huggingface web

2. Via Ollama repo

- Transform model to GGUF
- Quantize to 4bit

3. Load in Ollama

- Create Modelfile
- Create model
