import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yfinance as yf
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# -------------------------------------------------
# -------------------------------------------------
# Perceptron
# for Linear Regression
# -------------------------------------------------
# -------------------------------------------------

# The code provided is an example of a time series prediction task,
# where a simple linear regression model is used to predict
# the closing stock prices of Microsoft (MSFT),
# based on the closing prices of Apple (AAPL).

# -------------------------------------------------
# Source: ChatGPT - GPT-4
# -------------------------------------------------


# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 10
batch_size = 32
learning_rate = 0.001

training_eval_split_ratio = 0.7


# -------------------
# Data
# -------------------
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
split = int(0.8 * len(X))
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
# Model - Single layer Perceptron
# -------------------
class SLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(SLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, output_size))

    def forward(self, x):
        return self.layers(x)


model = SLP(input_size, output_size)


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
# # Use mean squared error loss for regression task
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # lower learning rate

# # Training
# epochs = 200
# losses = []
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()

#     y_pred = model(X_train)

#     loss = criterion(y_pred, y_train)
#     losses.append(loss.item())
#     loss.backward()

#     optimizer.step()

#     if (epoch + 1) % 50 == 0:
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# # plot training loss
# plt.figure()
# plt.plot(range(epochs), losses)
# plt.xlabel("Epoch")
# plt.ylabel("Training Loss")
# plt.title("Training Loss over time")
# plt.savefig("training_loss.png")

# # -------------------
# # Evaluation
# # -------------------
# model.eval()
# y_pred_test = model(X_test)
# loss_test = criterion(y_pred_test, y_test)
# print(f"Test Loss: {loss_test.item():.4f}")

# # plot actual vs predicted prices
# plt.figure()
# plt.plot(y_test.detach().numpy(), label="Actual MSFT prices")
# plt.plot(y_pred_test.detach().numpy(), label="Predicted MSFT prices")
# plt.xlabel("Time step")
# plt.ylabel("Price")
# plt.title("Actual vs Predicted MSFT Prices")
# plt.legend()
# plt.savefig("evaluation.png")
