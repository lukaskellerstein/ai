import os
import json
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Function Implementations
def get_stock_price(ticker: str):
    ticker_info = yf.Ticker(ticker).info
    current_price = ticker_info.get("currentPrice")
    return {"ticker": ticker, "current_price": current_price}


def get_dividend_date(ticker: str):
    ticker_info = yf.Ticker(ticker).info
    dividend_date = ticker_info.get("dividendDate")
    return {"ticker": ticker, "dividend_date": dividend_date}

# Define custom tools
tools = [
     {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Use this function to get the current price of a stock.",
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
        }
    },
     {
        "type": "function",
            "function": {
            "name": "get_dividend_date",
            "description": "Use this function to get the next dividend payment date of a stock.",
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
        }
    },
]

# Function to process messages and handle function calls
def get_completion_from_messages(messages, model="gpt-4o", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,  # Custom tools
        tool_choice="auto",  # Allow AI to decide if a tool should be called
        temperature=temperature,
    )

    response_message = response.choices[0].message

    print("First response:", response_message)

    if response_message.tool_calls:
        available_functions = {
            "get_stock_price": get_stock_price,
            "get_dividend_date": get_dividend_date,
        }

        function_name = response_message.tool_calls[0].function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message.tool_calls[0].function.arguments)
        function_response = function_to_call(**function_args)

        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": response_message.tool_calls[0].id,  
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": response_message.tool_calls[0].function.arguments,
                    }
                }
            ]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": response_message.tool_calls[0].id,  
            "name": function_name,
            "content": json.dumps(function_response),
        })

        # Second call to get final response based on function output
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        final_answer = second_response.choices[0].message

        print("Second response:", final_answer)
        return final_answer

    return "No relevant function call found."

# Example usage
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the current stock price for MSFT?"},
]

response = get_completion_from_messages(messages, temperature=1)
print(response)
