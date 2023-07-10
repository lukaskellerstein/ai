import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import sklearn.preprocessing
import numpy as np

start = time.time()


# -------------------
# -------------------
# 1. Get data
# -------------------
# -------------------

df = pd.read_csv("TSLA.csv", index_col=0)

print(df)


plt.plot(df.open.values, color="red", label="open")
plt.plot(df.close.values, color="green", label="close")
plt.plot(df.low.values, color="blue", label="low")
plt.plot(df.high.values, color="black", label="high")
plt.title("stock price")
plt.xlabel("time [minutes]")
plt.ylabel("price")
plt.legend(loc="best")

plt.show()

min_max_scaler = sklearn.preprocessing.MinMaxScaler()
df["open"] = min_max_scaler.fit_transform(df.open.values.reshape(-1, 1))
df["high"] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
df["low"] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
df["close"] = min_max_scaler.fit_transform(df["close"].values.reshape(-1, 1))
data = df[["open", "close", "low", "high"]].values

# -------------------
# Preparing data
# -------------------
seq_len = 20
sequences = []
for index in range(len(data) - seq_len):
    sequences.append(data[index : index + seq_len])
sequences = np.array(sequences)

# -------------------
# Splitting data
# -------------------
valid_set_size_percentage = 10
test_set_size_percentage = 10

valid_set_size = int(np.round(valid_set_size_percentage / 100 * sequences.shape[0]))
test_set_size = int(np.round(test_set_size_percentage / 100 * sequences.shape[0]))
train_set_size = sequences.shape[0] - (valid_set_size + test_set_size)

x_train = sequences[:train_set_size, :-1, :]
y_train = sequences[:train_set_size, -1, :]

x_valid = sequences[train_set_size : train_set_size + valid_set_size, :-1, :]
y_valid = sequences[train_set_size : train_set_size + valid_set_size, -1, :]

x_test = sequences[train_set_size + valid_set_size :, :-1, :]
y_test = sequences[train_set_size + valid_set_size :, -1, :]


# -------------------
# Creating data loaders
# -------------------
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()

x_valid = torch.tensor(x_valid).float()
y_valid = torch.tensor(y_valid).float()

train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = TensorDataset(x_valid, y_valid)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)


# -------------------
# -------------------
# 2. Define the NN architecture
# -------------------
# -------------------


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(4, 64, batch_first=True)
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        x = self.fc(hidden)
        return x


model = NeuralNetwork()

# push to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# -------------------
# -------------------
# 3. Train the network
# -------------------
# -------------------
optimizer = optim.Adam(model.parameters())
mse = nn.MSELoss()


def train(dataloader):
    epoch_loss = 0
    model.train()

    for batch in dataloader:
        optimizer.zero_grad()
        x, y = batch
        pred = model(x)
        loss = mse(pred[0], y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # return cumulative loss for whole epoch
    return epoch_loss


def evaluate(dataloader):
    epoch_loss = 0
    model.eval()

    # no_grad() prevents updating weights !!!
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            pred = model(x)
            loss = mse(pred[0], y)
            epoch_loss += loss.item()

    # return the average loss for whole epoch
    return epoch_loss / len(dataloader)


n_epochs = 10
best_valid_loss = float("inf")

for epoch in range(n_epochs):
    train_loss = train(train_dataloader)
    valid_loss = evaluate(valid_dataloader)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model, "saved_weights.pt")

    print("Epoch ", epoch + 1)
    print(f"\tTrain Loss: {train_loss:.5f}")
    print(f"\tVal Loss: {valid_loss:.5f}\n")


# -------------------
# -------------------
# 4. Test the trained network
# -------------------
# -------------------

model = torch.load("saved_weights.pt")

x_test = torch.tensor(x_test).float()

with torch.no_grad():
    y_test_pred = model(x_test)

y_test_pred = y_test_pred.numpy()[0]

idx = 0
plt.plot(
    np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]),
    y_test[:, idx],
    color="black",
    label="test target",
)

plt.plot(
    np.arange(y_train.shape[0], y_train.shape[0] + y_test_pred.shape[0]),
    y_test_pred[:, idx],
    color="green",
    label="test prediction",
)

plt.title("future stock prices")
plt.xlabel("time [days]")
plt.ylabel("normalized price")
plt.legend(loc="best")

plt.show()


end = time.time()
print(f"NN takes: {end - start} sec.")
