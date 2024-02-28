import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_type = os.getenv("API_TYPE")
openai.api_key = os.getenv("API_KEY")
openai.api_base = os.getenv("API_BASE")
openai.api_version = os.getenv("API_VERSION")

def get_completion(prompt, model="gpt-35-turbo-16k-deployment"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        deployment_id=model,
        messages=messages,
        temperature=1,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


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
response = get_completion(prompt)
print(response)
