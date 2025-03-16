import requests
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

api_key = os.environ.get("OPENAI_API_KEY")

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}


# CUSTOM TOOLS
tools = [
    {
        "type": "function",
        "function": {
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
    },
    {
        "type": "function",
        "function": {
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
    },
]

data = {
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the expected return for GOOG?"},
    ],
    "tools": tools,
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions", headers=headers, json=data
)
print(response.json())
