import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from helpers import plot_to_image

writer = SummaryWriter()

# -------------------
# Data
# -------------------

# Download historical market data
hist = yf.Ticker("AAPL").history(period="10y")
prices = hist[["Open", "High", "Low", "Close"]].values

# Calculate the daily price change
price_changes = prices[1:] - prices[:-1]

# Labels: 1 if price goes up the next day, 0 otherwise
labels = (prices[1:, 3] - prices[:-1, 3] > 0).astype(int)

# Split data into training, validation, and test sets
train_ratio = 0.6
valid_ratio = 0.2
train_split = round(price_changes.shape[0] * train_ratio)
valid_split = round(price_changes.shape[0] * (train_ratio + valid_ratio))
X_train, X_valid, X_test = (
    price_changes[:train_split],
    price_changes[train_split:valid_split],
    price_changes[valid_split:],
)
y_train, y_valid, y_test = (
    labels[:train_split],
    labels[train_split:valid_split],
    labels[valid_split:],
)

# Convert to PyTorch tensors
X_train, X_valid, X_test = (
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(X_valid, dtype=torch.float32),
    torch.tensor(X_test, dtype=torch.float32),
)
y_train, y_valid, y_test = (
    torch.tensor(y_train, dtype=torch.float32),
    torch.tensor(y_valid, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32),
)

train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)
test_dataset = TensorDataset(X_test, y_test)

# -------------------
# Visualizations
# -------------------

# Time series plot of prices
figure = plt.figure(figsize=(14, 7))
plt.plot(hist[["Open", "High", "Low", "Close"]])
plt.title("AAPL Stock Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(["Open", "High", "Low", "Close"])
plt.grid(True)
image = plot_to_image(figure)
writer.add_image("AAPL Stock Prices", image)

# Histogram of daily price changes
figure = plt.figure(figsize=(14, 7))
plt.hist(price_changes, bins=50, alpha=0.75)
plt.title("Histogram of Daily Price Changes")
plt.xlabel("Price Change")
plt.ylabel("Frequency")
plt.grid(True)
image = plot_to_image(figure)
writer.add_image("Histogram of Daily Price Changes", image)

# Pie chart for labels
figure = plt.figure(figsize=(7, 7))
plt.pie(
    pd.Series(labels).value_counts(),
    labels=["Price down or unchanged", "Price up"],
    autopct="%1.1f%%",
    startangle=140,
)
plt.title("Pie Chart of Labels")
image = plot_to_image(figure)
writer.add_image("Pie Chart of Labels", image)

# Box plot for daily price changes
figure = plt.figure(figsize=(14, 7))
plt.boxplot(price_changes, vert=False)
plt.title("Box Plot of Daily Price Changes")
plt.xlabel("Price Change")
plt.grid(True)
image = plot_to_image(figure)
writer.add_image("Box Plot of Daily Price Changes", image)

# Correlation heatmap
figure = plt.figure(figsize=(7, 7))
sns.heatmap(
    hist[["Open", "High", "Low", "Close"]].corr(), annot=True, cmap="coolwarm", center=0
)
plt.title("Correlation Heatmap")
image = plot_to_image(figure)
writer.add_image("Correlation Heatmap", image)
