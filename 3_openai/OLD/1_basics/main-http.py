import requests
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

api_key = os.environ.get("OPENAI_API_KEY")

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
```
"""

data = {
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions", headers=headers, json=data
)
print(response.json())
