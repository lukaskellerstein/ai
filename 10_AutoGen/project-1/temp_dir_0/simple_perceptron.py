# filename: simple_perceptron.py

import torch
import torch.nn as nn
import torch.optim as optim

# Define the Perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        return torch.sigmoid(self.fc(x_in)).squeeze()

# Create a simple dataset
# Input data
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# Output data
y_data = torch.tensor([[0.0], [0.0], [1.0], [1.0]]).squeeze()  # Squeeze to make it 1D

# Initialize the model
input_dim = 1
model = Perceptron(input_dim=input_dim)

# Define loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
for epoch in range(1000):
    # Forward pass
    y_pred = model(x_data)

    # Compute loss
    loss = loss_fn(y_pred, y_data)

    # Zero gradients, backward pass, update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Print out the loss after training
print(f'Loss after training: {loss.item()}')

# Test the model
test_input = torch.tensor([2.5])
test_output = model(test_input)
print(f'Prediction for input {test_input.item()}: {test_output.item()}')