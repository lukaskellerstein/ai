import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_run import test_dataset, test_data
from model import MLP
from train_test import run_testing
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from plots import plot

writer = SummaryWriter()

# -------------------------------------------------
# -------------------------------------------------
# Multi Layer Perceptron
# for Text Classification
# -------------------------------------------------
# -------------------------------------------------

# Hyper-parameters
input_size = 300  # this should match the length of your vectorized text
hidden_size = 200
output_size = 1
batch_size = 32

# Data
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = MLP(input_size, hidden_size, output_size)

# Loss function
loss_fn = nn.BCELoss()

# -------------------------------------------------
# -------------------------------------------------
# TEST = testing model on unseen data
# -------------------------------------------------
# -------------------------------------------------
model.load_state_dict(torch.load("saved_weights.pt"))

test_loss, test_acc, predicted_values = run_testing(model, test_loader, loss_fn)


# -------------------
# Plotting results - Testing
# -------------------
print(f"Test Loss: {test_loss:.4f}\n")
print(f"Test Accuracy: {test_acc:.4f}")

# Plotting test loss and accuracy
writer.add_scalar("Loss/test", test_loss)
writer.add_scalar("Accuracy/test", test_acc)


# Real vs. Predicted
y_test_final = []
for index, row in test_data.iterrows():
    y_test_final.append(row[1])

y_test_final = np.array(y_test_final)


final_predicted = []
for i in predicted_values:
    if i > 0.5:
        final_predicted.append(1)
    else:
        final_predicted.append(0)


plot_data = {
    "real_price": y_test_final,
    "predicted_price": final_predicted,
}
figure = plot(plot_data, "Real and Predicted values", "Count", "Value")
writer.add_figure("TEST/Real and Predicted values", figure)

# Diff
plot_data = {
    "diff": final_predicted - y_test_final,
}
figure = plot(
    plot_data, "Difference between Real and Predicted values", "Count", "Diff"
)
writer.add_figure("TEST/Difference between Real and Predicted values", figure)

writer.close()
