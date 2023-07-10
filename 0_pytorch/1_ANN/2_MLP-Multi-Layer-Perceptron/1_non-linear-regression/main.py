import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import yfinance as yf

yf.pdr_override()


# -------------------------------------------------
# -------------------------------------------------
# Multi Layer Perceptron
# for Non-Linear Regression
# -------------------------------------------------
# -------------------------------------------------

# The Multi-Layer Perceptron (MLP) is considered a type of non-linear regression model.

# In a nutshell, non-linear regression is a form of regression analysis
# in which observational data are modeled by a function which is a non-linear combination
# of the model parameters and depends on one or more independent variables.

# In our case, the MLP model learns a non-linear function that maps from
# the input data (stock prices at a given time)
# to the output data (stock prices at a future time).
# This function is determined by the parameters of the MLP
# (the weights and biases of the neurons),
# and these parameters are learned from the data via the training process.

# The non-linearity in the model comes from the activation function
# applied in the hidden layers, in this case, the ReLU (Rectified Linear Unit) function.


# -------------------------------------------------
# Source: ChatGPT - GPT-4
# -------------------------------------------------


# The code provided is an example of a time series prediction task,
# where a simple linear regression model is used to predict
# the closing stock prices of Microsoft (MSFT),
# based on the closing prices of Apple (AAPL).


# Hyper-parameters
input_size = 1
hidden_size = 64
output_size = 1
num_epochs = 10
batch_size = 32
learning_rate = 0.001

training_eval_split_ratio = 0.7

# -------------------
# Data
# -------------------
# Download historical data for desired stocks
# Get stock data
data = yf.download(["AAPL", "MSFT"], start="2022-01-01", end="2023-01-01")

# Prepare data
AAPL_prices = torch.tensor(data["Close"]["AAPL"].values).float()
MSFT_prices = torch.tensor(data["Close"]["MSFT"].values).float()

# Normalize data
AAPL_prices = (AAPL_prices - torch.mean(AAPL_prices)) / torch.std(AAPL_prices)
MSFT_prices = (MSFT_prices - torch.mean(MSFT_prices)) / torch.std(MSFT_prices)

# We will use AAPL prices to predict MSFT prices
X = AAPL_prices.view(-1, 1)
y = MSFT_prices.view(-1, 1)

# Split data into training and test sets (80% training, 20% test)
split = int(training_eval_split_ratio * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# Prepare dataset
class StockDataset(Dataset):
    def __init__(self, stock1, stock2):
        self.stock1 = stock1
        self.stock2 = stock2

    def __len__(self):
        return len(self.stock1)

    def __getitem__(self, idx):
        return self.stock1[idx], self.stock2[idx]


train_dataset = StockDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = StockDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -------------------
# Model - Multi layer Perceptron
# -------------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


model = MLP(input_size, hidden_size, output_size)

# -------------------------------------------------
# -------------------------------------------------
# EPOCHS - Training + Evaluation
# -------------------------------------------------
# -------------------------------------------------
# Loss function
criterion = nn.MSELoss()

# Finding optimal Loss function = Stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training function ------------
def train(dataloader):
    model.train()
    epoch_loss = 0

    n_total_steps = len(dataloader)

    for i, (stock1, stock2) in enumerate(dataloader):
        stock1 = Variable(stock1.view(-1, 1).float())
        stock2 = Variable(stock2.view(-1, 1).float())

        # Forward pass
        outputs = model(stock1)
        loss = criterion(outputs, stock2)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # print(f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

    # return average loss for whole epoch
    return epoch_loss / len(dataloader)


# Evaluate function ------------
def evaluate(dataloader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (stock1, stock2) in enumerate(dataloader):
            stock1 = Variable(stock1.view(-1, 1).float())
            stock2 = Variable(stock2.view(-1, 1).float())

            outputs = model(stock1)
            loss = criterion(outputs, stock2)
            epoch_loss += loss.item()

    # return the average loss for whole epoch
    return epoch_loss / len(dataloader)


# -------------------
# EPOCHS CYCLE
# -------------------
train_losses = []
eval_losses = []
best_valid_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = train(train_loader)
    valid_loss = evaluate(test_loader)

    # save losses
    train_losses.append(train_loss)
    eval_losses.append(valid_loss)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model, "saved_weights.pt")

    print("Epoch ", epoch + 1)
    print(f"\tTrain Loss: {train_loss:.5f}")
    print(f"\tVal Loss: {valid_loss:.5f}\n")

# -------------------
# Plotting results
# -------------------
# Plot of the training losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training loss")
plt.title("Training Losses")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_losses.png")

# Plot of the evaluation accuracies
plt.figure(figsize=(10, 5))
plt.plot(eval_losses, label="Test accuracy")
plt.title("Test Accuracies")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("evaluate_losses.png")


# # -------------------
# # Training
# # -------------------
# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Train model
# epochs = 100
# losses = []
# for epoch in range(epochs):
#     epoch_loss = 0
#     count = 0
#     for i, (stock1, stock2) in enumerate(train_dataloader):
#         stock1 = Variable(stock1.view(-1, 1).float())
#         stock2 = Variable(stock2.view(-1, 1).float())

#         # Forward pass
#         outputs = model(stock1)
#         loss = criterion(outputs, stock2)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()
#         count += 1

#     epoch_loss /= count
#     losses.append(epoch_loss)
#     print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, epochs, epoch_loss))

# # Plot the loss per epoch
# plt.figure(figsize=(14, 8))
# plt.plot(range(epochs), losses)
# plt.title("Loss over epochs")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.savefig("training_loss.png")


# # -------------------
# # Evaluation
# # -------------------
# model.eval()  # Set model to evaluation mode
# with torch.no_grad():
#     total_test_loss = 0
#     count = 0
#     for i, (stock1, stock2) in enumerate(test_dataloader):
#         stock1 = Variable(stock1.view(-1, 1).float())
#         stock2 = Variable(stock2.view(-1, 1).float())
#         outputs = model(stock1)
#         loss = criterion(outputs, stock2)
#         total_test_loss += loss.item()
#         count += 1

# print("Average test loss: {:.4f}".format(total_test_loss / count))

# # Plot the stocks and predicted correlation
# plt.figure(figsize=(14, 8))
# sort_idx = torch.argsort(
#     X_test.view(-1), dim=0
# )  # Sorting for a clear visual line for predictions
# X_test_sorted = X_test.view(-1)[sort_idx]
# y_test_sorted = y_test.view(-1)[sort_idx]
# plt.plot(X_test_sorted, y_test_sorted, "go", label="True data", alpha=0.5)
# predicted = model(X_test_sorted.view(-1, 1)).detach().view(-1)
# plt.plot(
#     X_test_sorted.numpy(), predicted.numpy(), "ro", label="Predicted data", alpha=0.5
# )
# plt.legend()
# plt.savefig("evaluation.png")
