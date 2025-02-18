import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import prepare_data
from model import MLP
from train_test import run_training
import argparse


# -----------------------------
# Azure
# -----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

print(args.data)
print(args.output_dir)

print("------------ START ------------")

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
train_dataset, valid_dataset = prepare_data(args.data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# -------------------
# Model - Multi layer Perceptron
# -------------------
print("model created")

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
print("start training")

results = run_training(
    num_epochs, model, train_loader, valid_loader, loss_fn, optimizer, patience
)


print("------------ ALL DONE ------------")
