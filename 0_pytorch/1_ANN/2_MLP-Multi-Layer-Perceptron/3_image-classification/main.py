import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import train_dataset, valid_dataset, test_dataset
from model import MLP
from epochs import run_epochs, test
from torch.utils.tensorboard import SummaryWriter
from helpers import plot_to_image

writer = SummaryWriter()

start = time.time()

# -------------------------------------------------
# -------------------------------------------------
# Multi Layer Perceptron
# for Image Classification
# -------------------------------------------------
# -------------------------------------------------


# Hyper-parameters
input_size = 784  # 28x28 images flattened into a 784 dimensional vector
hidden_size = 500
output_size = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001
patience = 5

# -------------------
# Data
# -------------------

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------
# Model - Multi layer Perceptron
# -------------------
model = MLP(input_size, hidden_size, output_size)


# Fetching a single batch from the train_loader
X_train, _ = next(iter(train_loader))

# Adding model structure to Tensorboard
writer.add_graph(model, X_train[0].unsqueeze(0))


# -------------------------------------------------
# -------------------------------------------------
# Training + Evaluation
# -------------------------------------------------
# -------------------------------------------------
# Loss function
loss_fn = nn.CrossEntropyLoss()

# Finding optimal Loss function = Stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# -------------------
# EPOCHS CYCLE
# -------------------
results = run_epochs(
    num_epochs, model, train_loader, valid_loader, loss_fn, optimizer, patience
)

# Adding results from epochs to Tensorboard
for epoch in range(results["last_epoch"]):
    writer.add_scalar("Loss/train", results["train_losses"][epoch], epoch)
    writer.add_scalar("Accuracy/train", results["train_accuracy"][epoch], epoch)
    writer.add_scalar("Loss/eval", results["valid_losses"][epoch], epoch)
    writer.add_scalar("Accuracy/eval", results["valid_accuracy"][epoch], epoch)

# -------------------
# Plotting results
# -------------------
figure = plt.figure(figsize=(10, 5))
plt.plot(results["train_losses"], label="Training loss")
plt.plot(results["valid_losses"], label="Validation loss")
plt.title("Training and Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
image = plot_to_image(figure)
writer.add_image("Loss", image, num_epochs)

figure = plt.figure(figsize=(10, 5))
plt.plot(results["train_accuracy"], label="Training accuracy")
plt.plot(results["valid_accuracy"], label="Validation accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
image = plot_to_image(figure)
writer.add_image("Accuracy", image, num_epochs)


# -------------------------------------------------
# -------------------------------------------------
# TEST = testing model on unseen data
# -------------------------------------------------
# -------------------------------------------------
model.load_state_dict(torch.load("saved_weights.pt"))

test_loss, test_acc, predicted_values, actual_values = test(model, test_loader, loss_fn)

print(f"Test Loss: {test_loss:.4f}\n")
print(f"Test Accuracy: {test_acc:.4f}")

# Plotting test loss and accuracy
for i in range(results["last_epoch"]):
    writer.add_scalar("Loss/test", test_loss, i)
    writer.add_scalar("Accuracy/test", test_acc, i)

# Plotting predicted and actual binary outcomes
figure = plt.figure(figsize=(10, 5))
plt.plot(predicted_values, label="Predicted values")
plt.plot(actual_values, label="Actual values")
plt.title("Predicted and Actual values on Test Set")
plt.xlabel("Count")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
image = plot_to_image(figure)
writer.add_image("Predictions vs Actual values", image, num_epochs)


# Plotting difference between predicted and actual binary outcomes
figure = plt.figure(figsize=(10, 5))
plt.plot(
    predicted_values - actual_values, label="Difference between Predicted and Actual"
)
plt.title("Difference between Predicted and Actual Binary Outcomes on Test Set")
plt.xlabel("Prediction count")
plt.ylabel("Difference")
plt.legend()
plt.grid(True)
image = plot_to_image(figure)
writer.add_image("Difference between Predicted and Actual values", image, num_epochs)

writer.close()


end = time.time()
print(f"NN takes: {end - start} sec.")
