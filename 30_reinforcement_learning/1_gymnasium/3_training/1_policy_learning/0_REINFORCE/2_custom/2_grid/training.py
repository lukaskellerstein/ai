import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from model import PolicyNetwork
import setup 

# Create environment
env = gym.make("Grid2DEnv-v0")

# =============================
# Hyperparameters
# =============================
INPUT_DIM = 2       # (row, col) position in the grid
OUTPUT_DIM = 4      # 4 possible actions (up, down, left, right)
HIDDEN_DIM = 128    # Hidden layer size in the neural network
LEARNING_RATE = 0.01  # Learning rate for the optimizer
GAMMA = 0.99        # Discount factor for rewards
NUM_EPISODES = 1000 # Number of training episodes
LOG_INTERVAL = 100  # Print training progress every N episodes

# Function to compute discounted rewards
def compute_discounted_rewards(rewards, gamma=GAMMA):
    discounted_rewards = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        discounted_rewards.insert(0, G)

    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

    # Normalize only if std() is nonzero
    if discounted_rewards.std() > 0:
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
    
    return discounted_rewards


# Training loop
def train(env, policy_net, optimizer, num_episodes=NUM_EPISODES, gamma=GAMMA):
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Ensure correct shape

        log_probs = []
        rewards = []
        done = False

        while not done:
            log_action_probs = policy_net(state)  # Now output is log-probs
            action_probs = torch.exp(log_action_probs)  # Convert to actual probabilities

            if torch.isnan(action_probs).any():
                print("Warning: NaN detected in action probabilities!")
                action_probs = torch.nan_to_num(action_probs, nan=1e-4)  # Replace NaNs with small value

            distribution = Categorical(action_probs)
            action = distribution.sample()
            log_probs.append(log_action_probs[:, action])  # Save log-prob directly from output

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(reward)
            
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # Ensure correct shape

        # Compute discounted rewards
        discounted_rewards = compute_discounted_rewards(rewards, gamma)

        # Compute policy loss
        loss = -torch.sum(torch.stack(log_probs) * discounted_rewards)  # Vectorized loss calculation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if episode % LOG_INTERVAL == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards):.2f}")

    # Save trained model
    torch.save(policy_net.state_dict(), "reinforce_policy.pth")
    print("Model saved successfully.")


if __name__ == "__main__":
    policy_net = PolicyNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    train(env, policy_net, optimizer)
