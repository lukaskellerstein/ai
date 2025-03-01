import gymnasium as gym
import torch
import numpy as np
from model import DPOPolicyNetwork
import setup

# Create environment
env = gym.make("Grid2DEnv-v0")

# =============================
# Hyperparameters
# =============================
INPUT_DIM = 2       # (row, col) position in the grid
OUTPUT_DIM = 4      # 4 possible actions (up, down, left, right)
HIDDEN_DIM = 128    # Hidden layer size in the neural network
MAX_STEPS = 50      # Maximum steps for inference

# Load the trained model and run inference
def test(env, policy_net, max_steps=MAX_STEPS):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        with torch.no_grad():
            action_probs = policy_net(state.unsqueeze(0))
            action = torch.argmax(action_probs).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        env.render()

        done = terminated or truncated
        state = torch.tensor(next_state, dtype=torch.float32)
        step_count += 1

    if done:
        print("Goal reached!")
    else:
        print("Max steps reached. Test ended.")

if __name__ == "__main__":
    policy_net = DPOPolicyNetwork(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM)
    policy_net.load_state_dict(torch.load("dpo_policy.pth"))
    policy_net.eval()  # Set to evaluation mode

    test(env, policy_net)
