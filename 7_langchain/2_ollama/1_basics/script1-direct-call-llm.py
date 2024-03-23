import os
from langchain_community.llms import Ollama

# ---------------------------
# Text Completion
# Ollama - Llama2 (basic)
# ---------------------------

llm = Ollama(model="mistral")
text = "What would be a good company name for a company that makes colorful socks?"
print(llm.invoke(text))

# OR
# CHAIN IS NOT POSSIBLE WITHOUT A PROMPT TEMPLATE
# chain = LLMChain(llm=llm, prompt=text)
# result = chain.run("colorful socks")
# print(result)


# curl http://localhost:11434/api/generate -d '{ \
#   "model": "llama2", \
#   "prompt": "Why is the sky blue?" \
# }'


# curl http://localhost:11434/api/generate -d '{"model": "llama2", "prompt": "Why is the sky blue?"}'
