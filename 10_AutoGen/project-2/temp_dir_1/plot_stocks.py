# filename: plot_stocks.py

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Download the stock data
msft_data = yf.download('MSFT', start=pd.to_datetime('today') - pd.DateOffset(years=1), end=pd.to_datetime('today'))
aapl_data = yf.download('AAPL', start=pd.to_datetime('today') - pd.DateOffset(years=1), end=pd.to_datetime('today'))

# Plot the MSFT data
plt.figure(figsize=(14, 7))
plt.plot(msft_data['Close'])
plt.title('MSFT Stock Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.savefig('msft.png')

# Plot the AAPL data
plt.figure(figsize=(14, 7))
plt.plot(aapl_data['Close'])
plt.title('AAPL Stock Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.savefig('aapl.png')

# Calculate the correlation between the MSFT and AAPL stock prices
correlation = msft_data['Close'].corr(aapl_data['Close'])

# Plot the correlation
plt.figure(figsize=(14, 7))
plt.scatter(msft_data['Close'], aapl_data['Close'])
plt.title('Correlation between MSFT and AAPL Stock Prices')
plt.xlabel('MSFT Price ($)')
plt.ylabel('AAPL Price ($)')
plt.grid(True)
plt.text(min(msft_data['Close']), max(aapl_data['Close']), f'Correlation: {correlation:.2f}', fontsize=12)
plt.savefig('correlation.png')