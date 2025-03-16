import os
from pprint import pprint
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv() 

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

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

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "You are an AI assistant."},
        {
            "role": "user",
            "content": "What is the expected return for GOOG?",
        },
    ],
    tools=tools # CUSTOM TOOLS
)

print("--- Full response: ---")
print(response.to_json())
print("--- Response text: ---")
print(response.choices[0].message.content)
print("--- Response Tool call: ---")
print(response.choices[0].message.tool_calls)