import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PPOActorCritic, self).__init__()

        # Shared feature extraction
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Actor (Policy network)
        self.actor_fc = nn.Linear(hidden_dim, output_dim)

        # Critic (Value function)
        self.critic_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        # Actor: Returns action probabilities
        action_probs = F.softmax(self.actor_fc(x), dim=-1)

        # Critic: Returns state-value (scalar)
        value = self.critic_fc(x)

        return action_probs, value
