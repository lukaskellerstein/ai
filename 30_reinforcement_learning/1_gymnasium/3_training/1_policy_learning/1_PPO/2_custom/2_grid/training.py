import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from model import PPOActorCritic
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
GAMMA = 0.99           # Discount factor for rewards
EPS_CLIP = 0.2         # PPO clipping range
BATCH_SIZE = 5         # Mini-batch size for PPO updates
UPDATE_EPOCHS = 4      # Number of optimization epochs per batch
NUM_EPISODES = 1000    # Number of training episodes
LOG_INTERVAL = 100     # Print training progress every N episodes

# Function to compute discounted rewards
def compute_discounted_rewards(rewards, gamma=GAMMA):
    discounted_rewards = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        discounted_rewards.insert(0, G)

    return torch.tensor(discounted_rewards, dtype=torch.float32)

# Training loop
def train(env, policy, optimizer, num_episodes=NUM_EPISODES, gamma=GAMMA):
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        log_probs = []
        values = []
        rewards = []
        done = False

        while not done:
            action_probs, value = policy(state)
            dist = Categorical(action_probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze())
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(reward)

            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # Compute discounted rewards
        discounted_rewards = compute_discounted_rewards(rewards, gamma)
        values = torch.stack(values)

        # Compute advantage
        advantages = discounted_rewards - values.detach()

        # Compute policy loss (PPO with clipped surrogate)
        old_log_probs = torch.stack(log_probs).detach()
        new_log_probs = torch.stack(log_probs)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Compute value loss
        value_loss = nn.MSELoss()(values, discounted_rewards)

        # Optimize
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        optimizer.step()

        # Logging
        if episode % LOG_INTERVAL == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards):.2f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

    # Save trained model
    torch.save(policy.state_dict(), "ppo_policy.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    policy_net = PPOActorCritic(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    train(env, policy_net, optimizer)
