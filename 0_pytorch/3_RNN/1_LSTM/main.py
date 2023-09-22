import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from plots import plot
from train_test import run_training, run_testing
from model import LSTM
from data import (
    train_dataset,
    valid_dataset,
    test_dataset,
    y_test_final,
    y_valid_final,
    min_max_scaler,
)


start = time.time()

writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# -------------------------------------------------
# -------------------------------------------------
# LSTM
# Long Short Term Memory
# -------------------------------------------------
# -------------------------------------------------

# Hyper-parameters
seq_len = 20

input_size = 1
num_layers = 1
hidden_size = 30
output_size = 1

num_epochs = 10
learning_rate = 0.001
patience = 5

# -------------------
# -------------------
# Data
# -------------------
# -------------------

train_loader = DataLoader(train_dataset, shuffle=False)
valid_loader = DataLoader(valid_dataset, shuffle=False)
test_loader = DataLoader(test_dataset, shuffle=False)

# -------------------
# -------------------
# Model
# -------------------
# -------------------
model = LSTM(input_size, hidden_size, num_layers, output_size)
model = model.to(device)

# -------------------------------------------------
# -------------------------------------------------
# Training + Evaluation
# -------------------------------------------------
# -------------------------------------------------
# Loss function
loss_fn = nn.MSELoss()

# Finding optimal Loss function = Stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


results = run_training(
    num_epochs, model, train_loader, valid_loader, loss_fn, optimizer, patience
)

# -------------------
# -------------------
# Testing
# -------------------
# -------------------
model.load_state_dict(torch.load("saved_weights.pt"))

test_loss, test_loss_rmse, predicted_values, actual_values = run_testing(
    model, test_loader, loss_fn
)

# -------------------
# -------------------
# Plotting results
# -------------------
# -------------------

# -------------------
# Plotting results - Training + Evaluation
# -------------------
# Adding results from epochs to Tensorboard
for epoch in range(results["last_epoch"]):
    writer.add_scalar("Loss-MSE/train", results["train_losses"][epoch], epoch)
    writer.add_scalar("Loss-RMSE/train", results["train_losses_rmse"][epoch], epoch)
    writer.add_scalar("Loss-MSE/eval", results["valid_losses"][epoch], epoch)
    writer.add_scalar("Loss-RMSE/eval", results["valid_losses_rmse"][epoch], epoch)

plot_data = {
    "train_loss": results["train_losses"],
    "valid_loss": results["valid_losses"],
}
figure = plot(plot_data, "Training and Validation Losses", "Epoch", "Loss")
writer.add_figure("TRAIN/Loss", figure)

valid_predictions = []
for i in range(len(results["valid_predictions"])):
    new_prediction = []

    for item in results["valid_predictions"][i]:
        new_prediction.append(item.cpu())

    valid_predictions.append(new_prediction)

valid_predictions = np.array(valid_predictions)

plot_data = {
    "real_price": y_valid_final,
}
for i in range(len(valid_predictions)):
    plot_data[f"predicted_price_{i}"] = valid_predictions[i]

figure = plot(plot_data, "Real and Predicted prices", "Time", "Price")
writer.add_figure("VALID/Norm", figure)


# -------------------
# Plotting results - Testing
# -------------------

print(f"Test Loss - MSE (Mean Square Error): {test_loss:.4f}\n")
print(f"Test Loss - RMSE (Root Mean Square Error)): {test_loss_rmse:.4f}\n")

# Plotting test loss and accuracy
for i in range(results["last_epoch"]):
    writer.add_scalar("Loss-MSE/test", test_loss, i)
    writer.add_scalar("Loss-RMSE/test", test_loss_rmse, i)


predictions = np.concatenate(
    [item.cpu().numpy().flatten() for item in predicted_values]
)
actuals = np.concatenate([item.cpu().numpy().flatten() for item in actual_values])

plot_data = {
    "original_price": y_test_final,
}
figure = plot(plot_data, "Original prices", "Time", "Price")
writer.add_figure("TEST/Origin", figure)


plot_data = {
    "real_price": actuals,
    "predicted_price": predictions,
}
figure = plot(plot_data, "Real and Predicted prices", "Time", "Price")
writer.add_figure("TEST/Norm/Prediction", figure)


plot_data = {
    "diff": predictions - actuals,
}
figure = plot(plot_data, "Diff between Real and Predicted prices", "Time", "Price")
writer.add_figure("TEST/Norm/Diff", figure)

# # Suppose 'scaler' is your MinMaxScaler object
predictions = min_max_scaler.inverse_transform([predictions]).flatten()
actuals = min_max_scaler.inverse_transform([actuals]).flatten()


plot_data = {
    "real_price": actuals,
    "predicted_price": predictions,
}
figure = plot(plot_data, "Real and Predicted prices", "Time", "Price")
writer.add_figure("TEST/Price/Prediction", figure)


plot_data = {
    "diff": predictions - actuals,
}
figure = plot(plot_data, "Diff between Real and Predicted prices", "Time", "Price")
writer.add_figure("TEST/Price/Diff", figure)


# --------------------------------------------
writer.close()

end = time.time()
print(f"NN takes: {end - start} sec.")
