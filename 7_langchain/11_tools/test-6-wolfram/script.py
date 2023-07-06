from dotenv import load_dotenv, find_dotenv
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

_ = load_dotenv(find_dotenv())  # read local .env file

wolfram = WolframAlphaAPIWrapper()

result = wolfram.run("What is 2x+5 = -3x + 7?")
print(result)

result = wolfram.run("What is fibonacci(100)?")
print(result)
