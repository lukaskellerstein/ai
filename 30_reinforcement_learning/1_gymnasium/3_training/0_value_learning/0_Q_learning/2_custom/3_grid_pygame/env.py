import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# Define colors
WHITE = (255, 255, 255)
PLAYER_COLOR = (0, 0, 255)
GOAL_COLOR = (255, 0, 0)

def get_heatmap_color(visits, max_visits):
    if max_visits == 0:
        return (255, 255, 255)  # White for no visits

    intensity = int(255 * (visits / max_visits))  # Normalize visit count
    intensity = max(0, min(intensity, 255))  # Ensure intensity stays within 0-255

    return (255, 255 - intensity, 50)  # Smooth gradient from yellow to red

class PygameGridEnv(gym.Env):
    def __init__(self, episode=0):
        super(PygameGridEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1279, shape=(2,), dtype=np.int32)

        # Set grid properties
        self.grid_width = 800
        self.grid_height = 600
        self.player_size = 25  # ✅ Define player size inside __init__
        self.cell_size = self.player_size  # ✅ Ensure cell size uses this variable
        self.grid_x = self.grid_width // self.cell_size
        self.grid_y = self.grid_height // self.cell_size

        # Set goal position
        self.goal_position = np.array([500, 500], dtype=np.int32)

        # Initialize state
        self.state = {
            "previous_position": None,
            "current_position": None,
        }

        # Initialize visit heatmap
        self.visit_counts = np.zeros((self.grid_x, self.grid_y), dtype=np.int32)

        # Track current episode
        self.episode = episode

        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((self.grid_width, self.grid_height))
        pygame.display.set_caption("Pygame Grid Environment")
        self.clock = pygame.time.Clock()

    def step(self, action):
        # Get current position
        x, y = self.state["current_position"]
        self.state["previous_position"] = self.state["current_position"]

        # Apply movement based on action
        if action == 0 and y > 0:  # Up
            y -= self.cell_size
        elif action == 1 and y < self.grid_height - self.cell_size:  # Down
            y += self.cell_size
        elif action == 2 and x > 0:  # Left
            x -= self.cell_size
        elif action == 3 and x < self.grid_width - self.cell_size:  # Right
            x += self.cell_size

        # Update state
        self.state["current_position"] = np.array([x, y], dtype=np.int32)

        # Update visit count heatmap
        grid_x, grid_y = x // self.cell_size, y // self.cell_size
        self.visit_counts[grid_x, grid_y] += 1

        # Calculate reward
        if np.array_equal(self.state["current_position"], self.goal_position):
            reward = 1.0
            terminated = True # Natural terminal state - Goal reached
            truncated = False # Not a timeout
            print(f"✅ Goal reached in {self.step_counter} steps!")
        elif self.step_counter >= self.max_steps:  # ✅ End if max steps reached
            reward = -0.5  # Mild penalty for running out of time
            terminated = False # Not a natural terminal state
            truncated = True  # Indicates timeout
            print(f"❌ Max steps ({self.max_steps}) reached. Ending episode.")
        else:
            reward = -0.01 # Small penalty for each step
            terminated = False # Not a natural terminal state
            truncated = False # Not a timeout

        info = {}

        # Render environment
        self.render()

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Increase the episode count every time reset is called
        self.episode += 1

        # Reset the visit heatmap for a new episode
        self.visit_counts.fill(0)

        # Randomly initialize the agent's position
        self.state = {
            "previous_position": None,
            "current_position": (
                np.random.randint(0, self.grid_x) * self.cell_size,
                np.random.randint(0, self.grid_y) * self.cell_size,
            ),
        }
        info = {}

        return self.state, info

    def render(self, mode="human"):
        # Fill the screen with white
        self.screen.fill(WHITE)

        # Draw the heatmap of visited positions
        max_visits = np.max(self.visit_counts) or 1  # Avoid division by zero
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                visits = self.visit_counts[x, y]
                if visits > 0:
                    color = get_heatmap_color(visits, max_visits)
                    pygame.draw.rect(
                        self.screen,
                        color,
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
                    )

        # Draw the player
        pygame.draw.rect(
            self.screen,
            PLAYER_COLOR,
            (*self.state["current_position"], self.cell_size, self.cell_size),
        )

        # Draw the goal
        pygame.draw.rect(
            self.screen,
            GOAL_COLOR,
            (*self.goal_position, self.cell_size, self.cell_size),
        )

        # Display step count and episode number
        font = pygame.font.SysFont("Arial", 20)
        step_text = font.render(f"Steps: {np.sum(self.visit_counts)}", True, (0, 0, 255))
        episode_text = font.render(f"Episode: {self.episode}", True, (0, 0, 255))
        self.screen.blit(step_text, (20, 20))
        self.screen.blit(episode_text, (20, 50))

        # Update display
        pygame.display.flip()

    def close(self):
        pygame.quit()
