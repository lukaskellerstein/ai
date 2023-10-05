# filename: plot_relative_returns.py

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Download the stock data
msft_data = yf.download('MSFT', start=pd.to_datetime('today') - pd.DateOffset(years=1), end=pd.to_datetime('today'))
aapl_data = yf.download('AAPL', start=pd.to_datetime('today') - pd.DateOffset(years=1), end=pd.to_datetime('today'))

# Calculate the relative returns
msft_returns = msft_data['Close'].pct_change()
aapl_returns = aapl_data['Close'].pct_change()

# Plot the MSFT relative returns
plt.figure(figsize=(14, 7))
plt.plot(msft_returns)
plt.title('MSFT Relative Returns')
plt.xlabel('Date')
plt.ylabel('Relative Return')
plt.grid(True)
plt.savefig('msft_returns.png')

# Plot the AAPL relative returns
plt.figure(figsize=(14, 7))
plt.plot(aapl_returns)
plt.title('AAPL Relative Returns')
plt.xlabel('Date')
plt.ylabel('Relative Return')
plt.grid(True)
plt.savefig('aapl_returns.png')

# Calculate the correlation between the MSFT and AAPL relative returns
correlation = msft_returns.corr(aapl_returns)

# Plot the correlation
plt.figure(figsize=(14, 7))
plt.scatter(msft_returns, aapl_returns)
plt.title('Correlation between MSFT and AAPL Relative Returns')
plt.xlabel('MSFT Relative Return')
plt.ylabel('AAPL Relative Return')
plt.grid(True)
plt.text(min(msft_returns), max(aapl_returns), f'Correlation: {correlation:.2f}', fontsize=12)
plt.savefig('returns_correlation.png')