import os
import json
import yfinance as yf
from pprint import pprint
import anthropic
from dotenv import load_dotenv
load_dotenv()  

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
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
        "name": "get_stock_price",
        "description": "Use this function to get the current price of a stock.",
        "input_schema": {
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
        "name": "get_dividend_date",
        "description": "Use this function to get the next dividend payment date of a stock.",
        "input_schema": {
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

# Define available functions
available_functions = {
    "get_stock_price": get_stock_price,
    "get_dividend_date": get_dividend_date,
}

# Function to process messages and handle function calls
def get_completion_from_messages(messages, model="claude-3-7-sonnet-20250219"):
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system="You are a helpful AI assistant.",
        messages=messages,
        tools=tools,
        tool_choice={"type": "auto"}
    )

    print("First response:", response)

    # Check if there's a tool call in the response
    has_tool_call = any(item.type == "tool_use" for item in response.content)
    
    if has_tool_call:
        # Find the tool call content
        tool_call = next(item for item in response.content if item.type == "tool_use")
        
        # Extract tool name and arguments
        function_name = tool_call.name
        function_args = tool_call.input  
        tool_id = tool_call.id
        
        # Call the function
        function_to_call = available_functions[function_name]
        function_response = function_to_call(**function_args)
        
        # Append the assistant message with the tool call
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": function_name,
                    "input": function_args
                }
            ]
        })
        
        # Append the tool result - using tool_use_id instead of tool_call_id
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,  # Using tool_use_id as per the error message
                    "content": json.dumps(function_response)
                }
            ]
        })
        
        # Second call to get final response based on function output
        second_response = client.messages.create(
            model=model,
            max_tokens=1024,
            system="You are a helpful AI assistant.",
            messages=messages,
        )
        
        return second_response
    
    return response

# Example usage
messages = [
    {"role": "user", "content": "What is the current stock price for MSFT?"},
]

response = get_completion_from_messages(messages)
print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
if response.content and hasattr(response.content[0], 'text'):
    print(response.content[0].text)
else:
    print("No text content in the response")