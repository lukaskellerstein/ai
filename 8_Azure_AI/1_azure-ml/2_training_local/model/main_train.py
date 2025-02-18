import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_train import train_dataset, valid_dataset, val_data
from model import MLP
from train_test import run_training
import numpy as np
from plots import plot
import os
from dotenv import load_dotenv, find_dotenv
from azureml.core import Workspace
from azureml.core.experiment import Experiment
from azureml.core.model import Model


_ = load_dotenv(find_dotenv())  # read local .env file

name = os.getenv("WORSPACE_NAME")
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RG")


# -------------------------------------------
# Workspace
# -------------------------------------------

ws = Workspace.get(
    name=name,
    subscription_id=subscription_id,
    resource_group=resource_group,
)

print(ws)

# new experiment
experiment = Experiment(workspace=ws, name="test-experiment")
run = experiment.start_logging()


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
num_epochs = 100
batch_size = 32
learning_rate = 0.001
patience = 5

# -------------------
# Data
# -------------------
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# -------------------
# Model - Multi layer Perceptron
# -------------------
model = MLP(input_size, hidden_size, output_size)


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
results = run_training(
    num_epochs, model, train_loader, valid_loader, loss_fn, optimizer, patience
)

# Adding results from epochs to Tensorboard
for epoch in range(results["last_epoch"]):
    run.log("Loss/train", results["train_losses"][epoch])
    run.log("Accuracy/train", results["train_accuracy"][epoch])
    run.log("Loss/eval", results["valid_losses"][epoch])
    run.log("Accuracy/eval", results["valid_accuracy"][epoch])


# -------------------
# Plotting results
# -------------------

# Loss
plot_data = {
    "train_loss": results["train_losses"],
    "valid_loss": results["valid_losses"],
}
figure = plot(plot_data, "Training and Validation Losses", "Epoch", "Loss")
run.log_image(name="TRAIN/Loss", plot=figure)

# Accuracy
plot_data = {
    "train_accuracy": results["train_accuracy"],
    "valid_accuracy": results["valid_accuracy"],
}
figure = plot(plot_data, "Training and Validation Accuracy", "Epoch", "Accuracy")
run.log_image(name="TRAIN/Accuracy", plot=figure)

# -------------------
# Plotting results - Training + Evaluation
# -------------------
y_valid_final = []
for index, row in val_data.iterrows():
    y_valid_final.append(row[1])

y_valid_final = np.array(y_valid_final)

plot_data = {
    "real_price": y_valid_final,
}
for i in range(len(results["valid_predictions"])):
    plot_data[f"predicted_price_{i}"] = results["valid_predictions"][i]


figure = plot(plot_data, "Real and Predicted prices", "Time", "Price")
run.log_image(name="VALID/Norm", plot=figure)


# Stop Azure AI experiment
run.complete()

# save model to Azure AI
model = Model.register(
    workspace=ws, model_path="saved_weights.pt", model_name="my-model-test"
)
