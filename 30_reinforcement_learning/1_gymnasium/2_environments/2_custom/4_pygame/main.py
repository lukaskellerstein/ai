import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

BLACK = (0, 0, 0)


class PygameGridEnv(gym.Env):

    all_moves = []
    current_step = 0

    def __init__(self):
        super(PygameGridEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = move up, 1 = move down, 2 = move left, 3 = move right
        self.action_space = spaces.Discrete(4)

        # Observations: the current position on the grid (x, y)
        self.observation_space = spaces.Box(
            low=0, high=1279, shape=(2,), dtype=np.int32
        )

        # Set the goal position
        self.goal_position = np.array([500, 500], dtype=np.int32)

        # Initialize the state
        self.state = {
            "previous_position": None,
            "current_position": None,
        }

        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Pygame Grid Environment")
        self.clock = pygame.time.Clock()

        # Define player and goal size
        self.player_size = 25
        self.goal_size = 25

    def step(self, action):
        # Implement logic for taking a step in the environment
        x, y = self.state["current_position"]

        self.state["previous_position"] = self.state["current_position"]

        newX, newY = x, y
        if action == 0 and y > 0:
            newY -= self.player_size
        elif action == 1 and y < 600 - self.player_size:
            newY += self.player_size
        elif action == 2 and x > 0:
            newX -= self.player_size
        elif action == 3 and x < 800 - self.player_size:
            newX += self.player_size

        self.state["current_position"] = np.array([newX, newY], dtype=np.int32)

        self.all_moves.append(self.state["current_position"])

        # Calculate reward
        if np.array_equal(self.state["current_position"], self.goal_position):
            reward = 1.0
            terminated = True
        else:
            reward = -0.01
            terminated = False

        truncated = False  # Not using truncated in this example

        info = {}

        # Print the grid at each step
        self.render()

        self.current_step += 1

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)  # Initialize the random number generator with the seed
        self.state = {
            "previous_position": None,
            "current_position": (
                np.random.randint(0, 800 // self.player_size) * self.player_size,
                np.random.randint(0, 600 // self.player_size) * self.player_size,
            ),
        }
        info = {}
        return self.state, info

    def render(self, mode="human"):
        # Render the environment using Pygame
        self.screen.fill((255, 255, 255))  # Fill the screen with white

        self.screen.blit(
            pygame.font.SysFont("Arial", 20).render(
                f"Step: {self.current_step}", True, (255, 0, 0)
            ),
            (20, 20),
        )

        # draw line of the movement from previous position to current position
        for i in range(1, len(self.all_moves)):
            pygame.draw.line(
                self.screen,
                BLACK,
                self.all_moves[i - 1],
                self.all_moves[i],
            )

        # Draw the player
        pygame.draw.rect(
            self.screen,
            (0, 0, 255),
            (*self.state["current_position"], self.player_size, self.player_size),
        )

        # Draw the goal
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            (*self.goal_position, self.goal_size, self.goal_size),
        )

        # Update the display
        pygame.display.update()
        self.clock.tick(30)

    def close(self):
        # Cleanup any resources used by the environment
        pygame.quit()


# Registering the Environment
from gymnasium.envs.registration import register

register(
    id="PygameGridEnv-v0",
    entry_point="__main__:PygameGridEnv",  # Adjust this to match your module and class name
)

# Using the Custom Environment
import gymnasium as gym

# Ensure the custom environment is registered
register(
    id="PygameGridEnv-v0",
    entry_point="__main__:PygameGridEnv",
)

# Create an instance of the custom environment
env = gym.make("PygameGridEnv-v0")

# Interact with the environment
state, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Sample a random action
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
