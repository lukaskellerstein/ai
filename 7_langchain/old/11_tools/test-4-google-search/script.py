from dotenv import load_dotenv, find_dotenv
from langchain.utilities import GoogleSerperAPIWrapper

_ = load_dotenv(find_dotenv())  # read local .env file


search = GoogleSerperAPIWrapper()

result = search.run("Obama's first name?")
print(result)
