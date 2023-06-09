import torch


class Perceptron(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train(model, train_inputs, train_labels, epochs=100, learning_rate=0.001):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_inputs)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()


def test(model, test_inputs, test_labels):
    output = model(test_inputs)
    correct = 0
    for i in range(len(test_labels)):
        if output[i] > 0.5:
            prediction = 1
        else:
            prediction = 0
        if prediction == test_labels[i]:
            correct += 1
    return correct / len(test_labels)


# Define the input and output data
X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.Tensor([[0], [1], [1], [0]])

# Create an instance of the Perceptron class
model = Perceptron(2)

# Train the model
train(model, X, y, epochs=1000)

# Test the model
accuracy = test(model, X, y)
print(f"Accuracy: {accuracy:.2f}")
