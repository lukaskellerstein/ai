from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


# CUSTOM FUNCTIONS ------------------------------------------------------------
functions = [
    {
        "name": "get_expected_returns_for_stock",
        "description": "Use this function to get the expected return for a stock. The output will be in JSON format.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The ticker symbol for the stock, e.g. GOOG",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_stock_price_target",
        "description": "Use this function to get the price target for a stock. The output will be in JSON format.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The ticker symbol for the stock, e.g. GOOG",
                }
            },
            "required": ["ticker"],
        },
    },
]


# ------------------------------------------------------------

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the expected return for GOOG?"},
]

# -----------------------------------------
# FIRST CALL -> to get what function to call
# -----------------------------------------
response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=messages,
    functions=functions,  # CUSTOM FUNCTIONS
    function_call="auto",
)
print(response)
