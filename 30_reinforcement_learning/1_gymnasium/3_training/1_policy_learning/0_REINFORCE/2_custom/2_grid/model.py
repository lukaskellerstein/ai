import torch
import torch.nn as nn
import torch.nn.functional as F  # Import functional API

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Hidden layer with ReLU
        return F.log_softmax(self.fc2(x), dim=-1)  # Use log-softmax instead of softmax
