from typing import Any
from semantic_kernel.functions import kernel_function


class StocksPlugin:
    """
    Description: Plugin for stocks.
    """

    @kernel_function(description="Get price for Stock ticker")
    def get_stock_price (self, ticker: str) -> str:
        return f"The price of {ticker} is $100"