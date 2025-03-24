import yfinance as yf

def get_dividend_date(ticker: str) -> dict:
    """Get the next dividend date of a stock.
    
    Args:
        ticker: The stock ticker symbol (e.g., AAPL, GOOG).
    """
    info = yf.Ticker(ticker).info
    return {"ticker": ticker, "dividend_date": info.get("dividendDate")}
