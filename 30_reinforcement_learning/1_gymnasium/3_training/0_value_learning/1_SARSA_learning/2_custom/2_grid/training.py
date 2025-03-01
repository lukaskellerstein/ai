import numpy as np
import gymnasium as gym
import setup

# Create environment
env = gym.make("Grid2DEnv-v0")

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000 # Number of training episodes

# Q-table: 5x5 grid with 4 possible actions
q_table = np.zeros((5, 5, 4))

def choose_action(state):
    """Epsilon-greedy action selection"""
    row, col = state
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Random action
    return np.argmax(q_table[row, col])  # Greedy action

def train_sarsa():
    """SARSA training loop (on-policy)"""
    for episode in range(episodes):
        state, _ = env.reset()
        action = choose_action(state)  # Choose first action
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = choose_action(next_state)  # Choose next action

            row, col = state
            next_row, next_col = next_state

            # ----------------------------------------------
            # SARSA update rule (on-policy)
            q_table[row, col, action] += alpha * (
                reward + gamma * q_table[next_row, next_col, next_action] - q_table[row, col, action]
            )

            state, action = next_state, next_action  # Move to next state and action
            # ----------------------------------------------
            
            done = terminated or truncated

        if episode % 100 == 0:
            print(f"SARSA: Episode {episode} completed")

    np.save("sarsa_q_table.npy", q_table)
    print("SARSA training complete. Q-table saved!")

if __name__ == "__main__":
    train_sarsa()
    env.close()
