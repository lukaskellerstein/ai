import torch.nn as nn


# -------------------
# Model - Single layer Perceptron
# -------------------
class SLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(SLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, output_size), nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)
