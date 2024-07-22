import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            2
        )  # Example: two possible actions (0 and 1)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # Example: one-dimensional observation between 0 and 1

        self.state = 0.5  # Initial state

    def step(self, action):
        # print(f"Action: {action}")

        # Implement logic for taking a step in the environment
        if action == 0:
            self.state -= 0.1
        else:
            self.state += 0.1

        # Ensure state stays within bounds
        self.state = np.clip(self.state, 0, 1)

        # Calculate reward (e.g., reward for being close to 0.5)
        reward = -abs(self.state - 0.5)

        # Check if episode is done
        terminated = bool(self.state == 0 or self.state == 1)
        truncated = False  # Set truncated to False since we're not handling time limits in this example

        # Optionally, you can add more information to the info dictionary
        info = {}

        return (
            np.array([self.state], dtype=np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)  # Initialize the random number generator with the seed
        self.state = 0.5
        info = {}  # Additional info dictionary
        return np.array([self.state], dtype=np.float32), info

    def render(self, mode="human"):
        # Render the environment (optional)
        print(f"State: {self.state}")

    def close(self):
        # Cleanup any resources used by the environment (optional)
        pass
