import yfinance as yf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------
# -------------------------------------------------
# Perceptron
# for Classification
# -------------------------------------------------
# -------------------------------------------------

# The model in the provided script is a simple binary classifier
# that aims to predict whether an email is spam (1) or not (0).

# The prediction is made based on various features of the email,
# such as the frequency of certain words or characters.
# Specifically, the model uses these features as inputs to learn
# a function that maps these characteristics to a binary outcome:
# 1 (the email is spam) or
# 0 (the email is not spam).

# Hyper-parameters
input_size = 57  # there are 57 features in the Spambase dataset
output_size = 1
num_epochs = 50
batch_size = 32
learning_rate = 0.001
patience = 5

# -------------------
# Data
# -------------------

# Load spambase dataset
data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
    header=None,
)

X = data.values[:, :-1]  # all columns except the last one
y = data.values[:, -1]  # only the last column

# Normalize data
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Split data into training, validation and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# Dataset
class SpamDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = SpamDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = SpamDataset(X_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

test_dataset = SpamDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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
        outputs = model(x)
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
            outputs = model(x)
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

    # save the best model
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

# Convert X_test to a tensor before feeding it to the model
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Get predictions
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).squeeze().numpy()

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
plt.plot(test_predictions - y_test, label="Difference between Predicted and Actual")
plt.title("Difference between Predicted and Actual Binary Outcomes on Test Set")
plt.xlabel("Prediction count")
plt.ylabel("Difference")
plt.legend()
plt.grid(True)
plt.savefig("predictions_difference.png")
