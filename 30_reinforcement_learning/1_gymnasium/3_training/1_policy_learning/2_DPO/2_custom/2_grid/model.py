import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DPOPolicyNetwork, self).__init__()

        # Policy network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)  # Output action probabilities
