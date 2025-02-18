import torch
import numpy as np
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from torch import nn
from pathlib import Path
import shutil
import logging


MODEL_DIR = "model/"


# accuracy measurement function
def accuracy(outputs, labels):
    preds = torch.round(outputs)  # round off to nearest integer
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# Training function ------------
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for i, (var1, var2) in enumerate(dataloader):
        # Forward pass
        outputs = model(var1)

        loss = loss_fn(outputs, var2)
        acc = accuracy(outputs, var2)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # return average loss for whole epoch (all steps)
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


# Evaluate function ------------
def evaluate(dataloader, model, loss_fn):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    predictions = np.array([])

    with torch.no_grad():
        for i, (var1, var2) in enumerate(dataloader):
            outputs = model(var1)

            predictions = np.append(predictions, outputs.detach().numpy().flatten())

            loss = loss_fn(outputs, var2)
            acc = accuracy(outputs, var2)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader), predictions


def run_training(
    num_epochs, model, train_loader, valid_loader, loss_fn, optimizer, patience
):
    train_losses = []
    train_accuracy = []
    valid_losses = []
    valid_accuracy = []
    valid_predictions = []
    best_valid_loss = float("inf")
    no_improve_epochs = 0  # check for overfitting
    last_epoch = 0

    for epoch in range(num_epochs):
        last_epoch = epoch

        train_loss, train_acc = train(train_loader, model, loss_fn, optimizer)
        valid_loss, valid_acc, valid_predict = evaluate(valid_loader, model, loss_fn)

        # save losses and accuracy
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accuracy.append(valid_acc)

        # save predictions
        valid_predictions.append(valid_predict)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "saved_weights.pt")
            save_model(MODEL_DIR, model)
            no_improve_epochs = 0  # reset no improve counter
        else:
            no_improve_epochs += 1  # increment counter if no improvement

        print("Epoch ", epoch + 1)
        print(f"\tTrain Loss: {train_loss:.5f}")
        print(f"\tValid Loss: {valid_loss:.5f}\n")

        metrics = {
            "training_loss": train_loss,
            "training_accuracy": train_acc,
            "validation_loss": valid_loss,
            "validation_accuracy": valid_acc,
        }
        mlflow.log_metrics(metrics, step=epoch)

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
        "train_accuracy": train_accuracy,
        "valid_losses": valid_losses,
        "valid_accuracy": valid_accuracy,
        "valid_predictions": valid_predictions,
    }


def run_testing(model, dataloader, loss_fn):
    # ----------------------
    # Evaluate the model on the test set
    # => output is the average loss for the whole test set
    # ----------------------
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    predictions = np.array([])

    with torch.no_grad():
        for i, (var1, var2) in enumerate(dataloader):
            outputs = model(var1)
            loss = loss_fn(outputs, var2)
            acc = accuracy(outputs, var2)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            predictions = np.append(predictions, outputs.detach().numpy().flatten())

    test_loss = epoch_loss / len(dataloader)
    test_acc = epoch_acc / len(dataloader)

    return test_loss, test_acc, predictions


def save_model(model_dir: str, model: nn.Module) -> None:
    """
    Saves the trained model.
    """
    input_schema = Schema([ColSpec(type="double", name=f"col_{i}") for i in range(784)])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    code_paths = ["model.py", "train_test.py"]
    full_code_paths = [
        Path(Path(__file__).parent, code_path) for code_path in code_paths
    ]

    shutil.rmtree(model_dir, ignore_errors=True)
    logging.info("Saving model to %s", model_dir)
    mlflow.pytorch.save_model(
        pytorch_model=model,
        path=model_dir,
        code_paths=full_code_paths,
        signature=signature,
    )
