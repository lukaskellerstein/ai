import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import setup
from model import DQN

# Create environment
env = gym.make("Grid2DEnv-v0")

# Hyperparameters
alpha = 0.001  # Learning rate for neural network
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000  # Number of training episodes
batch_size = 32  # Mini-batch size for experience replay
buffer_size = 5000  # Experience replay buffer size
target_update_freq = 50  # Frequency to update the target network

# Experience replay buffer
replay_buffer = deque(maxlen=buffer_size)



# Initialize networks
state_size = 2  # (row, col)
action_size = env.action_space.n

# Initialize model
q_network = DQN(state_size, action_size)
target_network = DQN(state_size, action_size)  # Target network for stability
target_network.load_state_dict(q_network.state_dict())  # Initialize with same weights
target_network.eval()  # Target network does not update during training

optimizer = optim.Adam(q_network.parameters(), lr=alpha)
loss_fn = nn.MSELoss()  # Mean squared error loss

def choose_action(state):
    """Epsilon-greedy action selection"""
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert to tensor
        with torch.no_grad():
            q_values = q_network(state_tensor)  # Get Q-values
        return torch.argmax(q_values).item()  # Select action with highest Q-value

def train_dqn():
    """DQN training loop with experience replay"""
    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state  # Move to next state

            # Train the network if we have enough experiences
            if len(replay_buffer) >= batch_size:
                train_step()

        # Update target network every few episodes
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        if episode % 100 == 0:
            print(f"DQN: Episode {episode} completed")

    # Save trained model
    torch.save(q_network.state_dict(), "dqn_model.pth")
    print("DQN training complete. Model saved!")

def train_step():
    """Sample a batch from replay buffer and train the network"""
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)  # Reshape for indexing
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.BoolTensor(dones)

    # Compute Q(s, a) from the current network
    q_values = q_network(states).gather(1, actions).squeeze()

    # Compute target Q-values using the target network
    with torch.no_grad():
        max_q_values = target_network(next_states).max(dim=1)[0]
        targets = rewards + gamma * max_q_values * (~dones)

    # Compute loss and update network
    loss = loss_fn(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    train_dqn()
    env.close()
