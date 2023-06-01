import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.getenv("OPENAI_API_KEY")


# ------------------------------------------------------------


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    #     print(str(response.choices[0].message))
    return response.choices[0].message["content"]


# ------------------------------------------------------------

# Example 1
messages = [
    {"role": "system", "content": "You are an assistant that speaks like Shakespeare."},
    {"role": "user", "content": "tell me a joke"},
    {"role": "assistant", "content": "Why did the chicken cross the road"},
    {"role": "user", "content": "I don't know"},
]

# Example 2
# messages =  [
# {'role':'system', 'content':'You are friendly chatbot.'},
# {'role':'user', 'content':'Hi, my name is Isa'}  ]

# Example 3
# messages =  [
# {'role':'system', 'content':'You are friendly chatbot.'},
# {'role':'user', 'content':'Yes,  can you remind me, What is my name?'}  ]

# Example 4
# messages =  [
# {'role':'system', 'content':'You are friendly chatbot.'},
# {'role':'user', 'content':'Hi, my name is Isa'},
# {'role':'assistant', 'content': "Hi Isa! It's nice to meet you. \
# Is there anything I can help you with today?"},
# {'role':'user', 'content':'Yes, you can remind me, What is my name?'}  ]


# ------------------------------------------------------------
response = get_completion_from_messages(messages, temperature=1)
print(response)
