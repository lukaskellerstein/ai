import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = torch.device('mps') # Apple’s Metal Performance Shaders (MPS)
# device = torch.device('cpu') # CPU


# accuracy measurement function
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# Training function ------------
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    # n_total_steps = len(dataloader)

    for i, (var1, var2) in enumerate(dataloader):
        var1 = var1.to(device)
        var2 = var2.to(device)

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

        # print(f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

    # return average loss for whole epoch (all steps)
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


# Evaluate function ------------
def evaluate(dataloader, model, loss_fn):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    # n_total_steps = len(dataloader)

    with torch.no_grad():
        for i, (var1, var2) in enumerate(dataloader):
            var1 = var1.to(device)
            var2 = var2.to(device)

            outputs = model(var1)
            loss = loss_fn(outputs, var2)
            acc = accuracy(outputs, var2)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            # print(f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

    # return the average loss for whole epoch (all steps)
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def run_epochs(
    num_epochs, model, train_loader, valid_loader, loss_fn, optimizer, patience
):
    train_losses = []
    train_accuracy = []
    valid_losses = []
    valid_accuracy = []
    best_valid_loss = float("inf")
    no_improve_epochs = 0  # check for overfitting
    last_epoch = 0

    for epoch in range(num_epochs):
        last_epoch = epoch

        train_loss, train_acc = train(train_loader, model, loss_fn, optimizer)
        valid_loss, valid_acc = evaluate(valid_loader, model, loss_fn)

        # Add histogram of model weights to Tensorboard
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        # save losses and accuracy
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accuracy.append(valid_acc)

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

    return {
        "last_epoch": last_epoch + 1,
        "train_losses": train_losses,
        "train_accuracy": train_accuracy,
        "valid_losses": valid_losses,
        "valid_accuracy": valid_accuracy,
    }


def test(model, dataloader, loss_fn):
    # ----------------------
    # Evaluate the model on the test set
    # => output is the average loss for the whole test set
    # ----------------------
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    # n_total_steps = len(dataloader)

    xxxx = torch.tensor([])
    yyyy = torch.tensor([])

    with torch.no_grad():
        for i, (var1, var2) in enumerate(dataloader):
            var1 = var1.to(device)
            var2 = var2.to(device)

            xxxx = torch.tensor([]).to(device)
            yyyy = torch.tensor([]).to(device)

            xxxx = torch.cat((xxxx, var1), 0)
            yyyy = torch.cat((yyyy, var2), 0)

            outputs = model(var1)
            loss = loss_fn(outputs, var2)
            acc = accuracy(outputs, var2)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            # print(f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

    test_loss = epoch_loss / len(dataloader)
    test_acc = epoch_acc / len(dataloader)

    # ----------------------
    # Get predictions on the test set
    # => output is all predictions
    # ----------------------

    model.eval()
    with torch.no_grad():
        test_predictions = torch.argmax(model(xxxx), dim=1)

    # test_predictions = predicted values
    # yyyy = actual values
    return test_loss, test_acc, test_predictions.cpu().numpy(), yyyy.cpu().numpy()
