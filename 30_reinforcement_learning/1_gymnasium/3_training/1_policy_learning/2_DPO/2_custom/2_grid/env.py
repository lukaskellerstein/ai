import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Grid2DEnv(gym.Env):
    def __init__(self):
        super(Grid2DEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = move up, 1 = move down, 2 = move left, 3 = move right
        self.action_space = spaces.Discrete(4)
        print("----------------------------")
        print("ACTION SPACE")
        print("0 (up) 1 (down) 2 (left) 3 (right)")
        print("----------------------------") 

        # Observations: the current position on the grid (row, column)
        self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32)
        print("----------------------------")
        print("OBSERVATION SPACE (Indexes)")
        for i in range(5):
            print(" ".join(f"{i},{j}" for j in range(5)))
        print("----------------------------") 

        # Set the goal position
        self.goal_position = np.array([4, 4], dtype=np.int32)

        # Initialize the state
        self.state = None
        self.step_counter = 0
        self.max_steps = 50

    def step(self, action):
        # Increment step counter
        self.step_counter += 1

        # Implement logic for taking a step in the environment
        row, col = self.state
        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and row < 4:
            row += 1
        elif action == 2 and col > 0:
            col -= 1
        elif action == 3 and col < 4:
            col += 1

        self.state = np.array([row, col], dtype=np.int32)

        action_text = ["up", "down", "left", "right"][action]
        print(f"Step: {self.step_counter} ---> Action: {action}={action_text}, State: {self.state}, Goal: {self.goal_position}")

        # Check if goal reached
        if np.array_equal(self.state, self.goal_position):
            reward = 1.0
            terminated = True
            print(f"✅ Goal reached in {self.step_counter} steps!")
        elif self.step_counter >= self.max_steps:  # ✅ End if max steps reached
            reward = -1.0  # Penalize for failing
            terminated = True
            print(f"❌ Max steps ({self.max_steps}) reached. Ending episode.")
        else:
            reward = -0.01  # Small penalty for each step
            terminated = False

        truncated = False  # Not using truncated in this example
        info = {}

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)  # Initialize the random number generator with the seed

        # Reset step counter
        self.step_counter = 0  
        
        # Randomly initialize the agent's position, avoiding the goal position
        while True:
            self.state = np.random.randint(0, 5, size=(2,), dtype=np.int32)
            if not np.array_equal(self.state, self.goal_position):
                break

        print("----------------------------")
        print("INITIAL STATE")
        self.render()
        print("----------------------------")                           

        info = {}
        return self.state, info

    def render(self, mode="human"):
        # Render the environment (optional)
        grid = [["-"] * 5 for _ in range(5)]
        row, col = self.state
        grid[row][col] = "A"  # Agent's position
        gr, gc = self.goal_position
        grid[gr][gc] = "G"  # Goal position
        for row in grid:
            print(" ".join(row))
        print()

    def close(self):
        # Cleanup any resources used by the environment (optional)
        pass
