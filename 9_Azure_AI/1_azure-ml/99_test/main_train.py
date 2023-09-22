import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_train import train_dataset, valid_dataset
from model import MLP
from train_test import run_training

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
