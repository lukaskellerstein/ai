# Copyright (c) 2017 Facebook, Inc. All rights reserved.
# BSD 3-Clause License
#
# Script adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# ==============================================================================

# imports
import os
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# TODO - add mlflow logging


# define network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 6 * 6, 120)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# define functions
def train(train_loader, model, criterion, optimizer, epoch, device, print_freq):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_freq == 0:  # print every print_freq mini-batches
            print(
                "Epoch %d: [%5d] loss: %.3f"
                % (epoch + 1, i + 1, running_loss / print_freq)
            )
            running_loss = 0.0


def main(args):
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define train and dataset DataLoaders
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=False, transform=transform
    )

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
    )

    model = Net().to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )

    # train the model
    for epoch in range(args.epochs):
        print("Starting epoch %d" % (epoch))
        model.train()
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            args.print_freq,
        )

    print("Finished Training")

    # log model
    # mlflow.pytorch.log_model(model, "model")
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model, os.path.join(args.model_dir, "model.pt"))
    # mlflow.pytorch.save_model(model, f"{args.model_dir}/model")


def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--data-dir", type=str, help="directory containing CIFAR-10 dataset"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./", help="output directory for model"
    )
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="mini batch size for each gpu/process",
    )
    parser.add_argument(
        "--workers",
        default=2,
        type=int,
        help="number of data loading workers for each gpu/process",
    )
    parser.add_argument(
        "--learning-rate", default=0.001, type=float, help="learning rate"
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument(
        "--print-freq",
        default=200,
        type=int,
        help="frequency of printing training statistics",
    )

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # call main function
    main(args)
