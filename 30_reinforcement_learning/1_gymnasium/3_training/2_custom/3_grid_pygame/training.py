import numpy as np
import gymnasium as gym
import setup

# Create environment
env = gym.make("PygameGridEnv-v0")

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration-exploitation trade-off
episodes = 1000  # Number of training episodes

# Initialize episode number manually
# env.unwrapped.episode = 1 

# Q-table: Discretizing the 800x600 space into grid cells (32x24 grid)
grid_size_x = env.screen.get_width() // env.player_size  # 800 / 25 = 32
grid_size_y = env.screen.get_height() // env.player_size  # 600 / 25 = 24
q_table = np.zeros((grid_size_x, grid_size_y, 4))

for episode in range(episodes):
    # env.unwrapped.episode = episode + 1
    state_dict, info = env.reset()
    x, y = state_dict["current_position"]
    state = (x // env.player_size, y // env.player_size)  # Convert to grid index
    done = False

    while not done:
        # Choose action (ε-greedy policy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore (random)
        else:
            action = np.argmax(q_table[state[0], state[1], :])  # Exploit (best known action)

        # Take action and observe new state & reward
        next_state_dict, reward, terminated, truncated, _ = env.step(action)
        next_x, next_y = next_state_dict["current_position"]
        next_state = (next_x // env.player_size, next_y // env.player_size)
        done = terminated or truncated

        # Update Q-value using Q-learning formula
        q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + alpha * (
            reward + gamma * np.max(q_table[next_state[0], next_state[1], :]) - q_table[state[0], state[1], action]
        )

        # Move to next state
        state = next_state

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes} completed")

# Save trained Q-table
np.save("q_table_pygame.npy", q_table)
print("Training completed and Q-table saved!")

env.close()
