import yfinance as yf

def get_stock_price(ticker: str) -> dict:
    """Get the current stock price.
    
    Args:
        ticker: The stock ticker symbol (e.g., AAPL, GOOG).
    """
    info = yf.Ticker(ticker).info
    return {"ticker": ticker, "current_price": info.get("currentPrice")}
