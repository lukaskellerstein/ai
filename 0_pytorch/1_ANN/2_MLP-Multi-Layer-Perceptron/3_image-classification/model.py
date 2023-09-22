import torch.nn as nn


# -------------------
# Model - Multi layer Perceptron
# -------------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # Flatten the input images from [28,28] to [784]
        x = x.view(x.size(0), -1)
        return self.layers(x)
