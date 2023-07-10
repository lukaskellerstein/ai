# Import necessary libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# -------------------------------------------------
# -------------------------------------------------
# Multi Layer Perceptron
# for Image CLASIFFICATION
# -------------------------------------------------
# -------------------------------------------------

# Hyper-parameters
input_size = 784  # 28x28 images flattened into a 784 dimensional vector
hidden_size = 500
num_classes = 10
num_epochs = 6
batch_size = 100
learning_rate = 0.001

# Device configuration (runs on GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Data
# -------------------

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
train_dataset.train_data.to(torch.device("cuda:0"))  # put data into GPU entirely
train_dataset.train_labels.to(torch.device("cuda:0"))

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)
test_dataset.train_data.to(torch.device("cuda:0"))  # put data into GPU entirely
test_dataset.train_labels.to(torch.device("cuda:0"))


# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# -------------------
# Model - Multi layer Perceptron
# -------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# -------------------------------------------------
# -------------------------------------------------
# EPOCHS - Training + Evaluation
# -------------------------------------------------
# -------------------------------------------------
# Loss function
criterion = nn.CrossEntropyLoss()

# Finding optimal Loss function = Stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training function ------------
def train(dataloader):
    model.train()
    epoch_loss = 0

    n_total_steps = len(dataloader)

    for i, (images, labels) in enumerate(dataloader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

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
        for images, labels in dataloader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
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
