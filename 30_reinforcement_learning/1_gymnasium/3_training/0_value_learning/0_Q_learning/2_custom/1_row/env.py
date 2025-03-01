import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = move left, 1 = move right
        self.action_space = spaces.Discrete(2)

        print("----------------------------")
        print("ACTION SPACE")
        print("0 (left) 1 (right)")
        print("----------------------------") 

        # Observations: the current position on the grid
        self.observation_space = spaces.Discrete(10)

        print("----------------------------")
        print("OBSERVATION SPACE (Indexes)")
        print(" ".join(f"{i}" for i in range(self.observation_space.n)))
        print("----------------------------") 

        # Set the goal position
        self.goal_position = 9

        # Initialize step counter
        self.step_counter = 0

    def step(self, action):
        # Increment step counter
        self.step_counter += 1

        # Implement logic for taking a step in the environment
        if action == 0:
            self.state = max(0, self.state - 1)
        elif action == 1:
            self.state = min(self.observation_space.n - 1, self.state + 1)

        action_text = "left" if action == 0 else "right"
        print(f"Step: {self.step_counter} ---> Action: {action}={action_text}, State: {self.state}, Goal: {self.goal_position}")

        # Calculate reward
        if self.state == self.goal_position:
            reward = 1.0
            terminated = True
            # Print summary when episode finishes
            print(f"Episode finished in {self.step_counter} steps")
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

        print("----------------------------")
        print("INITIAL STATE")
        self.render()
        print("----------------------------") 

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
