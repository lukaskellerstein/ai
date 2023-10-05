# filename: plot_stock.py

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Download the stock data
data = yf.download('MSFT', start=pd.to_datetime('today') - pd.DateOffset(years=1), end=pd.to_datetime('today'))

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(data['Close'])
plt.title('MSFT Stock Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.savefig('msft.png')