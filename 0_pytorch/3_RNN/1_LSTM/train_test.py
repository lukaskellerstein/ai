import time
from matplotlib.figure import Figure
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from plots import plot_time_series, plot
from helpers import denormalize_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()


# -------------------
# -------------------
# 3. Train the network
# -------------------
# -------------------
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = 0
    epoch_loss_rmse = 0

    for i, (var1, var2) in enumerate(dataloader):
        var1 = var1.to(device)
        var2 = var2.to(device)

        # Forward pass
        outputs = model(var1)
        loss = loss_fn(outputs.unsqueeze(0), var2)
        rmse_loss = torch.sqrt(loss)

        # if i == 0:
        #     print("--------------TRAIN----------------")
        #     print("var1: ", var1)
        #     print("var2: ", var2)
        #     print("Predicted: ", outputs.detach())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_loss_rmse += rmse_loss.item()

    # return average loss for whole epoch (all steps)
    return epoch_loss / len(dataloader), epoch_loss_rmse / len(dataloader)


def evaluate(dataloader, model, loss_fn):
    model.eval()
    epoch_loss = 0
    epoch_loss_rmse = 0
    predictions = []

    with torch.no_grad():
        for i, (var1, var2) in enumerate(dataloader):
            var1 = var1.to(device)
            var2 = var2.to(device)

            outputs = model(var1)
            loss = loss_fn(outputs.unsqueeze(0), var2)
            rmse_loss = torch.sqrt(loss)

            # if i == 0:
            #     print("--------------VALIDATE----------------")
            #     print("var1: ", var1)
            #     print("var2: ", var2)
            #     print("Predicted: ", outputs.detach())

            epoch_loss += loss.item()
            epoch_loss_rmse += rmse_loss.item()

            predictions.append(outputs.detach())

    # return the average loss for whole epoch (all steps)
    return epoch_loss / len(dataloader), epoch_loss_rmse / len(dataloader), predictions


def run_training(
    num_epochs, model, train_loader, valid_loader, loss_fn, optimizer, patience
):
    train_losses = []
    train_losses_rmse = []
    valid_losses = []
    valid_losses_rmse = []
    valid_predictions = []
    best_valid_loss = float("inf")
    no_improve_epochs = 0  # check for overfitting
    last_epoch = 0

    for epoch in range(num_epochs):
        last_epoch = epoch

        train_loss, train_loss_rmse = train(train_loader, model, loss_fn, optimizer)
        valid_loss, valid_loss_rmse, valid_predict = evaluate(
            valid_loader, model, loss_fn
        )

        # Add histogram of model weights to Tensorboard
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        # save losses and accuracy
        train_losses.append(train_loss)
        train_losses_rmse.append(train_loss_rmse)
        valid_losses.append(valid_loss)
        valid_losses_rmse.append(valid_loss_rmse)

        # save predictions
        valid_predictions.append(valid_predict)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "saved_weights.pt")
            no_improve_epochs = 0  # reset no improve counter
        else:
            no_improve_epochs += 1  # increment counter if no improvement

        # Early Stopping =
        # where you stop training when the validation loss
        # has not decreased for a certain number of epochs.
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        print(
            "Epoch:",
            str(epoch + 1),
            "Train loss:",
            str(train_loss),
            "Valid loss:",
            str(valid_loss),
        )

    return {
        "last_epoch": last_epoch + 1,
        "train_losses": train_losses,
        "train_losses_rmse": train_losses_rmse,
        "valid_losses": valid_losses,
        "valid_losses_rmse": valid_losses_rmse,
        "valid_predictions": valid_predictions,
    }


def run_testing(model, dataloader, loss_fn):
    model.eval()
    epoch_loss = 0
    epoch_loss_rmse = 0

    predictions = []
    actual = []

    with torch.no_grad():
        for i, (var1, var2) in enumerate(dataloader):
            var1 = var1.to(device)
            var2 = var2.to(device)

            outputs = model(var1)
            loss = loss_fn(outputs.unsqueeze(0), var2)
            rmse_loss = torch.sqrt(loss)

            epoch_loss += loss.item()
            epoch_loss_rmse += rmse_loss.item()

            # if i == 0:
            #     print("--------------TEST----------------")
            #     print("var1: ", var1)
            #     print("var2: ", var2)
            #     print("Predicted: ", outputs.detach())

            predictions.append(outputs.detach())
            actual.append(var2.detach())

    test_loss = epoch_loss / len(dataloader)
    test_loss_rmse = epoch_loss_rmse / len(dataloader)

    return test_loss, test_loss_rmse, predictions, actual
