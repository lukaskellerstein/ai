from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch

writer = SummaryWriter()

# -------------------
# Data
# -------------------

# MNIST TRAIN dataset
full_train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

# Split the full_train_dataset into train_dataset and valid_dataset
train_size = int(0.8 * len(full_train_dataset))  # 80% for training
valid_size = len(full_train_dataset) - train_size  # 20% for validation
train_dataset, valid_dataset = random_split(
    full_train_dataset, [train_size, valid_size]
)

# MNIST TEST dataset
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)


# -------------------
# TensorBoard Visualization
# -------------------

# Visualizing a few samples of the training data
images, labels = [], []

# Manually collect a batch of data
for i in range(64):  # We collect 64 samples as an example
    image, label = train_dataset[i]
    images.append(image)
    labels.append(label)

# Create grid of images
img_grid = torchvision.utils.make_grid(images)


# Write to tensorboard
writer.add_image("mnist_images", img_grid)

# Adding histogram for train, valid and test labels
train_labels = torch.tensor([label for _, label in train_dataset])  # Convert to tensor
valid_labels = torch.tensor([label for _, label in valid_dataset])  # Convert to tensor
test_labels = torch.tensor([label for _, label in test_dataset])  # Convert to tensor

writer.add_histogram("MNIST/train_labels", train_labels, bins="auto")
writer.add_histogram("MNIST/valid_labels", valid_labels, bins="auto")
writer.add_histogram("MNIST/test_labels", test_labels, bins="auto")

writer.close()
