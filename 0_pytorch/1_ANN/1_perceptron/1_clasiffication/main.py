import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# -------------------------------------------------
# -------------------------------------------------
# Perceptron
# for Classification
# -------------------------------------------------
# -------------------------------------------------


# The model in the provided script is a simple binary classifier
# that aims to predict whether the closing price of the Apple (AAPL)
# stock will go up (1) or down (0) the next day.

# The prediction is made based on the changes in
# the opening, closing, high, and low prices of the previous day.
# Specifically, the model uses these changes in prices as features
# (or inputs) to learn a function that maps these changes to
# a binary outcome:
# 1 (the stock price will go up) or
# 0 (the stock price will go down).

# Hyper-parameters
input_size = 4
output_size = 1
num_epochs = 100
batch_size = 32
learning_rate = 0.001
patience = 5


# -------------------
# Data
# -------------------

# Download historical market data
hist = yf.Ticker("AAPL").history(period="5y")
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

# Create data loaders
train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
)
valid_loader = DataLoader(
    TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
)


# -------------------
# Model - Single layer Perceptron
# -------------------
class SLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(SLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, output_size), nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


model = SLP(input_size, output_size)


# -------------------------------------------------
# -------------------------------------------------
# Training + Evaluation
# -------------------------------------------------
# -------------------------------------------------
# Loss function
loss_fn = nn.BCELoss()

# Finding optimal Loss function = Stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training function
def train(dataloader):
    model.train()
    epoch_loss = 0
    # n_total_steps = len(dataloader)

    for i, (x, y) in enumerate(dataloader):
        # Forward pass
        outputs = model(x).squeeze()
        loss = loss_fn(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # print(f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

    # return average loss for whole epoch (all steps)
    return epoch_loss / len(dataloader)


# Evaluation function
def evaluate(dataloader):
    model.eval()
    epoch_loss = 0
    # n_total_steps = len(dataloader)

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            # Forward pass
            outputs = model(x).squeeze()
            loss = loss_fn(outputs, y)

            epoch_loss += loss.item()
            # print(f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

    # return the average loss for whole epoch (all steps)
    return epoch_loss / len(dataloader)


# -------------------
# EPOCHS CYCLE
# -------------------
train_losses = []
valid_losses = []
best_valid_loss = float("inf")
no_improve_epochs = 0  # check for overfitting

for epoch in range(num_epochs):
    train_loss = train(train_loader)
    valid_loss = evaluate(valid_loader)

    # save losses
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "saved_weights.pt")
        no_improve_epochs = 0  # reset no improve counter
    else:
        no_improve_epochs += 1  # increment counter if no improvement

    print("Epoch ", epoch + 1)
    print(f"\tTrain Loss: {train_loss:.5f}")
    print(f"\tValid Loss: {valid_loss:.5f}\n")

    # Early Stopping =
    # where you stop training when the validation loss
    # has not decreased for a certain number of epochs.
    if no_improve_epochs >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break


# -------------------
# Plotting results
# -------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.title("Training and Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("losses.png")


# -------------------------------------------------
# -------------------------------------------------
# TEST = testing model on unseen data
# -------------------------------------------------
# -------------------------------------------------
model.load_state_dict(torch.load("saved_weights.pt"))


# ----------------------
# Evaluate the model on the test set
# => output is the average loss for the whole test set
# ----------------------
test_loss = evaluate(test_loader)
print(f"Test Loss: {test_loss:.4f}\n")

with open("test_loss.txt", "w") as f:
    f.write(f"Test Loss: {test_loss:.5f}\n")

# ----------------------
# Get predictions on the test set
# => output is all predictions
# ----------------------

# Get predictions
model.eval()
with torch.no_grad():
    test_predictions = model(X_test).squeeze().numpy()


# test_predictions = predicted values
# y_test = actual values


# Plotting predicted and actual binary outcomes
plt.figure(figsize=(10, 5))
plt.plot(test_predictions, label="Predicted values")
plt.plot(y_test, label="Actual values")
plt.title("Predicted and Actual values on Test Set")
plt.xlabel("Count")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig("predictions_vs_actual.png")

# Plotting difference between predicted and actual binary outcomes
plt.figure(figsize=(10, 5))
plt.plot(
    test_predictions - y_test.numpy(), label="Difference between Predicted and Actual"
)
plt.title("Difference between Predicted and Actual Binary Outcomes on Test Set")
plt.xlabel("Prediction count")
plt.ylabel("Difference")
plt.legend()
plt.grid(True)
plt.savefig("predictions_difference.png")
