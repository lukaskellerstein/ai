import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from model import DPOPolicyNetwork
import setup

# Create environment
env = gym.make("Grid2DEnv-v0")

# =============================
# Hyperparameters
# =============================
INPUT_DIM = 2          # (row, col) position in the grid
OUTPUT_DIM = 4         # 4 possible actions (up, down, left, right)
HIDDEN_DIM = 128       # Hidden layer size in the neural network
LEARNING_RATE = 0.002  # Learning rate for Adam optimizer
NUM_EPISODES = 1000    # Number of training episodes
LOG_INTERVAL = 100     # Print training progress every N episodes
BATCH_SIZE = 5         # Number of trajectory pairs used per update
BETA = 0.5             # DPO preference weight

# Function to collect preference data (trajectory pairs)
def collect_trajectory_pairs(env, policy, num_pairs=BATCH_SIZE):
    trajectory_pairs = []

    for _ in range(num_pairs):
        # Generate two trajectories
        traj_1 = generate_trajectory(env, policy)
        traj_2 = generate_trajectory(env, policy)

        # Preference: Choose trajectory closer to the goal
        preference = 1 if traj_1['reward'] > traj_2['reward'] else 0

        trajectory_pairs.append((traj_1, traj_2, preference))

    return trajectory_pairs

# Function to generate a single trajectory
def generate_trajectory(env, policy):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    log_probs = []
    rewards = []
    done = False
    total_reward = 0

    while not done:
        action_probs = policy(state)  # Compute action probabilities
        dist = Categorical(action_probs)

        action = dist.sample()
        log_prob = dist.log_prob(action).clone()  # Clone to avoid in-place modification
        log_probs.append(log_prob)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        rewards.append(reward)
        total_reward += reward

        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

    return {"log_probs": torch.stack(log_probs), "reward": total_reward}

# DPO Loss function (Fixed)
def dpo_loss(policy, traj_1, traj_2, preference, beta=BETA):
    log_pi_1 = traj_1["log_probs"].sum()  # No detach() here
    log_pi_2 = traj_2["log_probs"].sum()  # No detach() here

    diff = log_pi_1 - log_pi_2
    loss = -preference * diff + beta * torch.log1p(torch.exp(-diff))  # Use log1p(exp(x)) for numerical stability

    return loss.mean()

# Training loop
def train(env, policy, optimizer, num_episodes=NUM_EPISODES):
    for episode in range(num_episodes):
        trajectory_pairs = collect_trajectory_pairs(env, policy)

        # Accumulate the loss over all pairs
        total_loss = 0.0
        for traj_1, traj_2, preference in trajectory_pairs:
            total_loss += dpo_loss(policy, traj_1, traj_2, preference)

        # Average or scale the loss by the number of pairs
        total_loss = total_loss / len(trajectory_pairs)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Logging
        if episode % LOG_INTERVAL == 0:
            print(f"Episode {episode}, Avg Loss: {total_loss.item():.4f}")

    # Save trained model
    torch.save(policy.state_dict(), "dpo_policy.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    policy_net = DPOPolicyNetwork(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    train(env, policy_net, optimizer)
