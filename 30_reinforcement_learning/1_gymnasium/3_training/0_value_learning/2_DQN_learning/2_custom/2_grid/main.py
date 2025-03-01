import numpy as np
import gymnasium as gym
import torch
import setup
from model import DQN

# Create environment
env = gym.make("Grid2DEnv-v0")

# Load trained DQN model
state_size = 2  # (row, col)
action_size = 4  # Number of actions

# Initialize model
q_network = DQN(state_size, action_size)  
q_network.load_state_dict(torch.load("dqn_model.pth"))  # Load trained weights
q_network.eval()

def test():
    state, _ = env.reset()
    done = False

    print("\n=== Testing DQN Agent ===")
    print(f"Initial State: {state}")
    env.render()

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            # Choose best action
            action = torch.argmax(q_network(state_tensor)).item()  

        # Take action
        state, reward, terminated, truncated, _ = env.step(action)

        # Render environment
        env.render()

        # Stop if the episode ends
        done = terminated or truncated

    print("\n=== Testing Complete ===")
    env.close()

if __name__ == "__main__":
    test()
