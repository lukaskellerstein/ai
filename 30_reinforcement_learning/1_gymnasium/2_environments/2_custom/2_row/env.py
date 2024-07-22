import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = move left, 1 = move right
        self.action_space = spaces.Discrete(2)

        # Observations: the current position on the grid
        self.observation_space = spaces.Discrete(10)

        # Set the goal position
        self.goal_position = 9

    def step(self, action):
        print(f"Action: {action}")

        # Implement logic for taking a step in the environment
        if action == 0:
            self.state = max(0, self.state - 1)
        elif action == 1:
            self.state = min(self.observation_space.n - 1, self.state + 1)

        # Calculate reward
        if self.state == self.goal_position:
            reward = 1.0
            terminated = True
        else:
            reward = -0.01
            terminated = False

        truncated = False  # Not using truncated in this example

        info = {}

        self.render()

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)  # Initialize the random number generator with the seed
        self.state = np.random.randint(0, self.observation_space.n)
        info = {}
        return self.state, info

    def render(self, mode="human"):
        # Render the environment (optional)
        grid = ["-"] * self.observation_space.n
        grid[self.state] = "A"  # Agent's position
        grid[self.goal_position] = "G"  # Goal position
        print(" ".join(grid))

    def close(self):
        # Cleanup any resources used by the environment (optional)
        pass
