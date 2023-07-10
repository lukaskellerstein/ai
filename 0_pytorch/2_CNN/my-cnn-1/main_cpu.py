import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

start = time.time()


# -------------------------------------------------
# -------------------------------------------------
# CNN - Convolutional Neural Network
# for Image CLASIFFICATION
# -------------------------------------------------
# -------------------------------------------------


# Hyper-parameters
input_size = 1
hidden_size = 64
output_size = 1
num_epochs = 10
batch_size = 100
learning_rate = 0.001


# -------------------
# Data
# -------------------

# MNIST dataset
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# -------------------
# Model - CNN
# -------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 32, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Reshape before passing to the fully connected layer
        x = self.classifier(x)
        return x


model = CNN()


# -------------------------------------------------
# -------------------------------------------------
# EPOCHS - Training + Evaluation
# -------------------------------------------------
# -------------------------------------------------
# Loss function
criterion = nn.CrossEntropyLoss()

# Finding optimal Loss function = Stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training function ------------
def train(dataloader):
    model.train()
    epoch_loss = 0

    n_total_steps = len(dataloader)

    for i, (images, labels) in enumerate(dataloader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # print(f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

    # return average loss for whole epoch
    return epoch_loss / len(dataloader)


# Evaluate function ------------
def evaluate(dataloader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

    # return the average loss for whole epoch
    return epoch_loss / len(dataloader)


# -------------------
# EPOCHS CYCLE
# -------------------
train_losses = []
eval_losses = []
best_valid_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = train(train_loader)
    valid_loss = evaluate(test_loader)

    # save losses
    train_losses.append(train_loss)
    eval_losses.append(valid_loss)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model, "saved_weights.pt")

    print("Epoch ", epoch + 1)
    print(f"\tTrain Loss: {train_loss:.5f}")
    print(f"\tVal Loss: {valid_loss:.5f}\n")

# -------------------
# Plotting results
# -------------------
# Plot of the training losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training loss")
plt.title("Training Losses")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_losses.png")

# Plot of the evaluation accuracies
plt.figure(figsize=(10, 5))
plt.plot(eval_losses, label="Test accuracy")
plt.title("Test Accuracies")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("evaluate_losses.png")


end = time.time()
print(f"NN takes: {end - start} sec.")


# # -------------------
# # -------------------
# # 1. Get data
# # -------------------
# # -------------------
# batch_size = 64

# # Load the MNIST dataset
# train_dataset = datasets.MNIST(
#     root="data", train=True, transform=transforms.ToTensor(), download=True
# )
# test_dataset = datasets.MNIST(root="data", train=False, transform=transforms.ToTensor())
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# # -------------------
# # -------------------
# # 2. Define the NN architecture
# # -------------------
# # -------------------


# # Define the CNN model
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         self.fc1 = nn.Linear(7 * 7 * 32, 128)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = x.view(-1, 7 * 7 * 32)
#         x = self.relu3(self.fc1(x))
#         x = self.fc2(x)
#         return x


# # Set device (GPU or CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Initialize the model
# model = CNN().to(device)


# # -------------------
# # -------------------
# # 3. Train the network
# # -------------------
# # -------------------

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# num_epochs = 10

# # Train the model
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i + 1) % 100 == 0:
#             print(
#                 "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
#                     epoch + 1, num_epochs, i + 1, total_step, loss.item()
#                 )
#             )


# # -------------------
# # -------------------
# # 4. Test the trained network
# # -------------------
# # -------------------

# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print(
#         "Accuracy of the model on the test images: {:.2f}%".format(
#             100 * correct / total
#         )
#     )


# end = time.time()
# print(f"NN takes: {end - start} sec.")
