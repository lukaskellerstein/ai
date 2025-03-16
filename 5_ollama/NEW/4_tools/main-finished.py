import json
import yfinance as yf
from ollama import chat, ChatResponse

# Function Implementations
def get_stock_price(ticker: str):
    ticker_info = yf.Ticker(ticker).info
    current_price = ticker_info.get("currentPrice")
    return {"ticker": ticker, "current_price": current_price}


def get_dividend_date(ticker: str):
    ticker_info = yf.Ticker(ticker).info
    dividend_date = ticker_info.get("dividendDate")
    return {"ticker": ticker, "dividend_date": dividend_date}

# Define tools for Ollama
tools = [
    get_stock_price,
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
        },
    },
]

# Available functions for calling
available_functions = {
    "get_stock_price": get_stock_price,
    "get_dividend_date": get_dividend_date,
}

# Example usage
messages = [
    {"role": "user", "content": "What is the current stock price for MSFT?"}
]

response: ChatResponse = chat(
    "llama3.2",
    messages=messages,
    tools=tools,
)

print("First response:", response.message)

if response.message.tool_calls:

    # Find the tool call content
    tool_call = response.message.tool_calls[0]

    # Extract tool name and arguments
    function_name = tool_call.function.name
    function_args = tool_call.function.arguments

    # Call the function
    function_to_call = available_functions[function_name]
    function_response = function_to_call(**function_args)

    print("Function response:", function_response)

    # Add function response to messages for the model to use
    messages.append(response.message)
    messages.append({
        "role": "tool", 
        "name": function_name,
        "content": json.dumps(function_response), 
    })

    # Get final response from model
    final_response = chat("llama3.2", messages=messages)
    print("Second response:", final_response.message.content)
else:
    print("No tool calls returned from model")
