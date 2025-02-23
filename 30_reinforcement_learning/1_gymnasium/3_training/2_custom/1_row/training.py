import numpy as np
import gymnasium as gym
import setup

# Create environment
env = gym.make("CustomEnv-v0")

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration-exploitation trade-off
episodes = 500  # Number of training episodes

# Initialize Q-table with zeros
q_table = np.zeros((env.observation_space.n, env.action_space.n))

for episode in range(episodes):
    state, info = env.reset()
    done = False

    while not done:
        # Choose action (ε-greedy policy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore (random)
        else:
            action = np.argmax(q_table[state, :])  # Exploit (best known action)

        # Apply action and observe new state & reward
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update Q-value using the Q-learning formula
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )

        # Move to next state
        state = next_state

    # Print progress
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}/{episodes} completed")

env.close()

# Save trained Q-table
np.save("q_table.npy", q_table)
print("Training completed and Q-table saved!")
