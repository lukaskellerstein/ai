import os
import importlib
import json
import requests
import inspect
from termcolor import colored
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from transformers import AutoTokenizer

load_dotenv()  # This loads the variables from .env


model = "Trelis/Mistral-7B-Instruct-v0.2-function-calling-v3"
api_endpoint = "https://ax17kk73vpkws0jx.us-east-1.aws.endpoints.huggingface.cloud"
tgi_api_base = api_endpoint + "/generate"

## Use this for models that are fine-tuned for function calling
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.\n\n    Parameters:\n    - location (str): The city, e.g., 'San Francisco'.\n    - unit (str): The temperature unit to use, 'celsius' or 'fahrenheit'. Defaults to 'celsius'.\n\n    Returns:\n    - dict: A dictionary containing weather data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "default": "celsius"},
                },
                "required": ["location"],
            },
            "returns": "any",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_clothes",
            "description": "Function to recommend clothing based on temperature and weather condition.\n\n    Parameters:\n    - temperature (str): The temperature, e.g., '60 F' or '15 C'.\n    - condition (str): The weather condition, e.g., 'Sunny', 'Rainy'.\n\n    Returns:\n    - str: A string suggesting appropriate clothing for the given weather, or an error message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "string"},
                    "condition": {"type": "string"},
                },
                "required": ["temperature", "condition"],
            },
            "returns": "string",
        },
    },
]


# -------------------------------------
# Request 1
# -------------------------------------

messages = []

# Function Metadata
messages.append({"role": "function_metadata", "content": json.dumps(tools, indent=4)})

# User Prompt
messages.append({"role": "user", "content": "What is the current weather in London?"})


# Apply the chat template to the messages
formatted_messages = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(formatted_messages)

# Create the JSON payload
json_payload = {
    "inputs": formatted_messages,
}

print("--------- REQUEST --------------")
print(json_payload)
print("--------------------------------")

response = requests.post(
    tgi_api_base,
    json=json_payload,
    headers={"Content-Type": "application/json"},
)

response_data = response.json()

print("--------- RESPONSE -------------")
print(response_data)
print("--------------------------------")


messages.append(
    {
        "role": "function_call",
        "content": response_data["generated_text"],
    }
)

messages.append(
    {
        "role": "function_response",
        "content": {"temperature": "15 C", "condition": "Cloudy"},
    }
)

# -------------------------------------
# Request 2
# -------------------------------------

# Apply the chat template to the messages
formatted_messages = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(formatted_messages)

# Create the JSON payload
json_payload = {
    "inputs": formatted_messages,
}

print("--------- REQUEST --------------")
print(json_payload)
print("--------------------------------")

response = requests.post(
    tgi_api_base,
    json=json_payload,
    headers={"Content-Type": "application/json"},
)

response_data = response.json()

print("--------- RESPONSE -------------")
print(response_data)
print("--------------------------------")
