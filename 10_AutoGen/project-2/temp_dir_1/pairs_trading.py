# filename: pairs_trading.py

import yfinance as yf
import pandas as pd

# Download the stock data
msft_data = yf.download('MSFT', start=pd.to_datetime('today') - pd.DateOffset(years=1), end=pd.to_datetime('today'))
aapl_data = yf.download('AAPL', start=pd.to_datetime('today') - pd.DateOffset(years=1), end=pd.to_datetime('today'))

# Calculate the relative returns
msft_returns = msft_data['Close'].pct_change()
aapl_returns = aapl_data['Close'].pct_change()

# Calculate the spread between the two stocks
spread = msft_returns - aapl_returns

# Define the threshold for when we'll start trading
threshold = spread.mean() + spread.std()

# Create a DataFrame to hold our trades
trades = pd.DataFrame(index=spread.index)
trades['Long MSFT'] = spread > threshold
trades['Short MSFT'] = spread < -threshold
trades['No Trade'] = abs(spread) < threshold

# Calculate the profits from our trades
profits = -trades['Long MSFT'].diff().shift(-1).fillna(0) * msft_returns
profits += trades['Short MSFT'].diff().shift(-1).fillna(0) * msft_returns

# Print the total profit
print('Total profit:', profits.sum())