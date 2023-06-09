import openai
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")


# Embedding models
# each tuned to perform well on different functionalities:
# - text similarity,
# - text search
# - code search.
# The models take either text or code as input and return an embedding vector.

# MODEL
# text-embedding-ada-002
# replaces all old models. It can handle all tasks - text similarity, text search, and code search - and is the best model to use for all purposes.

# OLD MODELS
# text-similarity-{ada, babbage, curie, davinci}-001
# text-search-{ada, babbage, curie, davinci}-{query, doc}-001
# code-search-{ada, babbage}-{code, text}-001


text = """	
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""


response = openai.Embedding.create(input=text, engine="text-embedding-ada-002")
print(response)
