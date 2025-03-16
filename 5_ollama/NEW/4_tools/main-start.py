from pprint import pprint
from ollama import ChatResponse, chat

def get_stock_price(ticker: str) -> str:
    """
    Use this function to get the current price of a stock.
    """
    # Simulated response (replace with actual API call if needed)
    return f"The current stock price for {ticker} is $150."

tools = [
    get_stock_price,  # Function-based tool (Ollama supports direct function usage)
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

messages = [
    {"role": "user", "content": "What is the current stock price for MSFT?"},
]

response: ChatResponse = chat(
    "llama3.2", 
    messages=messages,
    tools=tools, # CUSTOM TOOLS
) 

print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
print(response.message.content)
print("--- Response Tool call: ---")
print(response.message.tool_calls)