import yfinance as yf
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("finance")

@mcp.tool()
def get_stock_price(ticker: str) -> dict:
    """Get the current stock price.
    
    Args:
        ticker: The stock ticker symbol (e.g., AAPL, GOOG).
    """
    info = yf.Ticker(ticker).info
    return {"ticker": ticker, "current_price": info.get("currentPrice")}

@mcp.tool()
def get_dividend_date(ticker: str) -> dict:
    """Get the next dividend date of a stock.
    
    Args:
        ticker: The stock ticker symbol (e.g., AAPL, GOOG).
    """
    info = yf.Ticker(ticker).info
    return {"ticker": ticker, "dividend_date": info.get("dividendDate")}

if __name__ == "__main__":
    mcp.run(transport='stdio')
