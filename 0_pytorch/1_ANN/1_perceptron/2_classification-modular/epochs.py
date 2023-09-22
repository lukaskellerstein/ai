import torch


# Training function ------------
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = 0
    # n_total_steps = len(dataloader)

    for i, (var1, var2) in enumerate(dataloader):
        # Forward pass
        outputs = model(var1)
        loss = loss_fn(outputs, var2)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # print(f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

    # return average loss for whole epoch (all steps)
    return epoch_loss / len(dataloader)


# Evaluate function ------------
def evaluate(dataloader, model, loss_fn):
    model.eval()
    epoch_loss = 0
    # n_total_steps = len(dataloader)

    with torch.no_grad():
        for i, (var1, var2) in enumerate(dataloader):
            outputs = model(var1)
            loss = loss_fn(outputs, var2)
            epoch_loss += loss.item()
            # print(f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

    # return the average loss for whole epoch (all steps)
    return epoch_loss / len(dataloader)


def run_epochs(
    num_epochs, model, train_loader, valid_loader, loss_fn, optimizer, patience
):
    train_losses = []
    valid_losses = []
    best_valid_loss = float("inf")
    no_improve_epochs = 0  # check for overfitting

    for epoch in range(num_epochs):
        train_loss = train(train_loader, model, loss_fn, optimizer)
        valid_loss = evaluate(valid_loader, model, loss_fn)

        # save losses
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "saved_weights.pt")
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

    return {"train_losses": train_losses, "valid_losses": valid_losses}


def test(model, test_loader, loss_fn):
    # ----------------------
    # Evaluate the model on the test set
    # => output is the average loss for the whole test set
    # ----------------------
    test_loss = evaluate(test_loader, model, loss_fn)
    print(f"Test Loss: {test_loss:.4f}\n")

    with open("test_loss.txt", "w") as f:
        f.write(f"Test Loss: {test_loss:.5f}\n")

    # ----------------------
    # Get predictions on the test set
    # => output is all predictions
    # ----------------------

    # Extract x and y from test set
    X_test_list = []
    y_test_list = []
    for batch in test_loader:
        X_batch, y_batch = batch
        X_test_list.append(X_batch)
        y_test_list.append(y_batch)

    # Concatenate all X tensors along the 0 dimension
    X_test_tensor = torch.cat(X_test_list, dim=0)

    # Concatenate all Y tensors along the 0 dimension
    y_test_loader = torch.cat(y_test_list, dim=0)
    # Convert tensor to numpy array
    y_test_loader = y_test_loader.numpy()
    # Flatten the array
    y_test_loader = y_test_loader.flatten()

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).squeeze().numpy()

    # test_predictions = predicted values
    # y_test = actual values
    return test_predictions, y_test_loader
