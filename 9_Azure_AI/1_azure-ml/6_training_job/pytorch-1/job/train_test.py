import torch
import numpy as np


# accuracy measurement function
def accuracy(outputs, labels):
    preds = torch.round(outputs)  # round off to nearest integer
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# Training function ------------
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    # n_total_steps = len(dataloader)

    for i, (var1, var2) in enumerate(dataloader):
        # Forward pass
        outputs = model(var1)

        # if i == 0:
        #     print("--------------TRAIN----------------")
        #     print("var1: ", var1)
        #     print("var2: ", var2)
        #     print("Predicted: ", outputs.detach())

        loss = loss_fn(outputs, var2)
        acc = accuracy(outputs, var2)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        # print(f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

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

            # if i == 0:
            #   print("--------------EVALUATE----------------")
            #   # print("var1: ", var1)
            #   # print("var2: ", var2)
            #   print("Predicted: ", outputs.detach().numpy().flatten())

            predictions = np.append(predictions, outputs.detach().numpy().flatten())

            loss = loss_fn(outputs, var2)
            acc = accuracy(outputs, var2)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    # return the average loss for whole epoch (all steps)
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
            # torch.save(model.state_dict(), "saved_weights.pt")
            no_improve_epochs = 0  # reset no improve counter
        else:
            no_improve_epochs += 1  # increment counter if no improvement

        print("Epoch ", epoch + 1)
        print(f"\tTrain Loss: {train_loss:.5f}")
        print(f"\tValid Loss: {valid_loss:.5f}\n")

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

    # xxxx = torch.tensor([])
    # yyyy = torch.tensor([])

    with torch.no_grad():
        for i, (var1, var2) in enumerate(dataloader):
            # xxxx = torch.cat((xxxx, var1), 0)
            # yyyy = torch.cat((yyyy, var2), 0)

            outputs = model(var1)
            loss = loss_fn(outputs, var2)
            acc = accuracy(outputs, var2)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            if i == 0:
                print("--------------TEST----------------")
                print("var1: ", var1)
                print("var2: ", var2)
                print("Predicted: ", outputs.detach().numpy().flatten())

            predictions = np.append(predictions, outputs.detach().numpy().flatten())

    test_loss = epoch_loss / len(dataloader)
    test_acc = epoch_acc / len(dataloader)

    # test_predictions = predicted values
    # yyyy = actual values
    return test_loss, test_acc, predictions
