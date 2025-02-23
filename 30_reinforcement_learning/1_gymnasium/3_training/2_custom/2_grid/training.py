import numpy as np
import gymnasium as gym
import setup

# Create environment
env = gym.make("Grid2DEnv-v0")

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration-exploitation trade-off
episodes = 1000  # Number of training episodes

# Q-table: 5x5 grid with 4 possible actions
q_table = np.zeros((5, 5, 4))

for episode in range(episodes):
    state, info = env.reset()
    done = False

    while not done:
        row, col = state  # Get current position

        # Choose action (ε-greedy policy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore (random)
        else:
            action = np.argmax(q_table[row, col, :])  # Exploit (best known action)

        # Take action and observe new state & reward
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_row, next_col = next_state
        done = terminated or truncated

        # Update Q-value using Q-learning formula
        q_table[row, col, action] = q_table[row, col, action] + alpha * (
            reward + gamma * np.max(q_table[next_row, next_col, :]) - q_table[row, col, action]
        )

        # Move to next state
        state = next_state

    # Print progress
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes} completed")

# Save trained Q-table
np.save("q_table_grid.npy", q_table)
print("Training completed and Q-table saved!")

env.close()
