import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -------------------
# Data Preparation
# -------------------
# Load the dataset
df = pd.read_csv("input_data/creditcard.csv")

# Split features and labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Assume that the majority of the data is 'normal' and anomalies are rare
# This is usually the case for anomaly detection problems
normal_data = X[y == 0].copy()
anomalous_data = X[y == 1].copy()

# Normalize the amount column
scaler = StandardScaler()
normal_data["Amount"] = scaler.fit_transform(
    normal_data["Amount"].values.reshape(-1, 1)
)

# Split the 'normal' data into training and validation sets
X_train, X_val = train_test_split(normal_data, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
X_test = torch.tensor(X.values, dtype=torch.float32)  # test on the entire dataset
y_test = torch.tensor(y.values, dtype=torch.float32)

# Create data loaders
train_loader = DataLoader(
    TensorDataset(X_train, torch.zeros(X_train.shape[0])), batch_size=32, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val, torch.zeros(X_val.shape[0])), batch_size=1, shuffle=False
)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1, shuffle=False)


# -------------------
# Model - Autoencoder
# -------------------
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder(X.shape[1], 14, 7)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training function
def train(dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        for i, (x, _) in enumerate(dataloader):
            output = model(x)
            loss = criterion(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


# Train the model
train(train_loader, epochs=50)  # increased the number of epochs

# Calculate the reconstruction error on the validation set
model.eval()
val_losses = []
with torch.no_grad():
    for i, (x, _) in enumerate(val_loader):
        output = model(x)
        loss = criterion(output, x)
        val_losses.append(loss.item())

# Set the threshold to be the 99th percentile of the validation losses
threshold = np.percentile(val_losses, 99)

# -------------------
# Anomaly Detection
# -------------------
# Calculate the reconstruction error on the test set and detect anomalies
model.eval()
test_losses = []
predictions = []
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        output = model(x)
        loss = criterion(output, x)
        test_losses.append(loss.item())
        predictions.append(
            1 if loss.item() > threshold else 0
        )  # 1 for anomalies, 0 for normal data

# Convert predictions to a tensor
predictions = torch.tensor(predictions)

# Plot the reconstruction errors, the threshold and the actual labels
plt.figure(figsize=(10, 5))
plt.plot(test_losses, "b")
plt.plot(
    np.where(predictions.numpy() == 1),
    np.array(test_losses)[predictions.numpy() == 1],
    "ro",
)
plt.axhline(y=threshold, color="r", linestyle="--")
plt.title("Reconstruction errors of test data and anomalies")
plt.xlabel("Example")
plt.ylabel("Reconstruction error")
plt.legend(["Reconstruction error", "Detected anomalies", "Threshold"])
plt.savefig("reconstruction_errors.png")

# Plot the actual labels and the detected anomalies
plt.figure(figsize=(10, 5))
plt.plot(y_test, "b")
plt.plot(predictions, "r")
plt.title("Actual labels and detected anomalies")
plt.xlabel("Example")
plt.ylabel("Label")
plt.legend(["Actual", "Prediction"])
plt.savefig("actual_labels_and_detected_anomalies.png")
