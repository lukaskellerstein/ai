import torch.nn as nn


# -------------------
# Model - CNN
# -------------------

# input_channels:
# The number of channels in the input image (1 for grayscale, 3 for RGB).

# conv_channels:
# A list of output channel sizes for each convolutional layer. For example, [16, 32] for your original model.

# kernel_size, stride, padding:
# Parameters for the convolutional layers.

# fc_hidden_size:
# The number of nodes in the hidden layer (fully connected).

# num_classes:
# The number of output classes (or the size of the output layer).


class CNN(nn.Module):
    def __init__(
        self,
        input_channels,
        conv_channels,
        kernel_size,
        stride,
        padding,
        fc_hidden_size,
        num_classes,
    ):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels,
                conv_channels[0],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                conv_channels[0],
                conv_channels[1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # We calculate the size of the flattened layer
        # We assume that the input is a 28x28 image, which is standard for MNIST
        flattened_size = (28 // (2 * 2)) ** 2 * conv_channels[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, fc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Reshape before passing to the fully connected layer
        x = self.classifier(x)
        return x
