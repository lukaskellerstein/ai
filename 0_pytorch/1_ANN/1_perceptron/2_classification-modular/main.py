import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import train_dataset, valid_dataset, test_dataset
from model import SLP
from epochs import run_epochs, test

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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -------------------
# Model - Single layer Perceptron
# -------------------
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


# -------------------
# EPOCHS CYCLE
# -------------------
losses = run_epochs(
    num_epochs, model, train_loader, valid_loader, loss_fn, optimizer, patience
)

# -------------------
# Plotting results
# -------------------
plt.figure(figsize=(10, 5))
plt.plot(losses["train_losses"], label="Training loss")
plt.plot(losses["valid_losses"], label="Validation loss")
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

predicted_values, actual_values = test(model, test_loader, loss_fn)

print("Predicted values: ", predicted_values)
print("Actual values: ", actual_values)

# Plotting predicted and actual binary outcomes
plt.figure(figsize=(10, 5))
plt.plot(predicted_values, label="Predicted values")
plt.plot(actual_values, label="Actual values")
plt.title("Predicted and Actual values on Test Set")
plt.xlabel("Count")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig("predictions_vs_actual.png")

# Plotting difference between predicted and actual binary outcomes
plt.figure(figsize=(10, 5))
plt.plot(
    predicted_values - actual_values, label="Difference between Predicted and Actual"
)
plt.title("Difference between Predicted and Actual Binary Outcomes on Test Set")
plt.xlabel("Prediction count")
plt.ylabel("Difference")
plt.legend()
plt.grid(True)
plt.savefig("predictions_difference.png")
