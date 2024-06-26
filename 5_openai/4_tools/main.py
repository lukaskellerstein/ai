import os
import openai
from dotenv import load_dotenv, find_dotenv
import yfinance as yf
import json

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_type = os.getenv("API_TYPE")
openai.api_key = os.getenv("API_KEY")
openai.api_base = os.getenv("API_BASE")
openai.api_version = os.getenv("API_VERSION")

# CUSTOM FUNCTIONS ------------------------------------------------------------


# Implementation
def get_expected_returns_for_stock(ticker: str):
    return {"ticker": ticker, "average_upside": "10%", "high_upside": "25%"}


def get_stock_price_target(ticker: str):
    ticker_info = yf.Ticker(ticker).info
    current_price = ticker_info.get("currentPrice")
    target_low = ticker_info.get("targetLowPrice")
    target_mid = ticker_info.get("targetMedianPrice")
    target_high = ticker_info.get("targetHighPrice")
    return {
        "current_price": current_price,
        "target_low": target_low,
        "target_mid": target_mid,
        "target_high": target_high,
    }


# OpenAI definition
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


def get_completion_from_messages(
    messages, model="gpt-35-turbo-16k-deployment", temperature=0
):

    # -----------------------------------------
    # FIRST CALL -> to get what function to call
    # -----------------------------------------
    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=temperature,
        functions=functions,  # CUSTOM FUNCTIONS
        function_call="auto",
    )

    response_message = response["choices"][0]["message"]

    print("first_response", response_message)

    if response_message.get("function_call"):
        available_functions = {
            "get_stock_price_target": get_stock_price_target,
            "get_expected_returns_for_stock": get_expected_returns_for_stock,
        }

        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(**function_args)
        function_response = str(function_response)

        # Add the assistant response and function response to the messages
        messages.append(  # adding assistant response to messages
            {
                "role": response_message["role"],
                "function_call": {
                    "name": function_name,
                    "arguments": response_message["function_call"]["arguments"],
                },
                "content": None,
            }
        )
        messages.append(  # adding function response to messages
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )

        # -----------------------------------------
        # SECOND CALL -> to answer the question
        # -----------------------------------------
        # Second API call to answer the users question based on the data retrieved from the custom function
        second_response = openai.ChatCompletion.create(
            engine=model,
            messages=messages,
        )
        answer = second_response["choices"][0]["message"]

        print("second_response", answer)

        return answer

    return "no answer"


# ------------------------------------------------------------

# Example 1
messages = [
    {"role": "system", "content": "You are helful assistant"},
    {"role": "user", "content": "What is expected return for TSLA stock?"},
]

# ------------------------------------------------------------
response = get_completion_from_messages(messages, temperature=1)
print(response)
