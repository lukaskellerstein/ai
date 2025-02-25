import gymnasium as gym
from gymnasium import spaces
import pyautogui
import numpy as np
import time

class ComputerControlEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(ComputerControlEnv, self).__init__()

        # Define action space: Move mouse, click, type keys
        self.action_space = spaces.Discrete(6)  # 0: Left, 1: Right, 2: Up, 3: Down, 4: Click, 5: Type

        # Action descriptions for logging
        self.action_descriptions = {
            0: "Mouse Move Left",
            1: "Mouse Move Right",
            2: "Mouse Move Up",
            3: "Mouse Move Down",
            4: "Mouse Click",
            5: "Type 'Hello'"
        }

        print("--------------------------------------------")
        print("ACTION SPACE")
        print("0 (Mouse Left) 1 (Mouse Right) 2 (Mouse Up) 3 (Mouse Down) 4 (Mouse Click) 5 (Keyboard Type 'Hello')")
        print("--------------------------------------------")

        # Observation space: Mouse position (x, y)
        screen_width, screen_height = pyautogui.size()
        print("Screen Size:", screen_width, screen_height)
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([screen_width, screen_height]), dtype=np.int32
        )

        print("----------------------------")
        print("OBSERVATION SPACE")
        print(self.observation_space)
        print("----------------------------")

        # Initialize step counter
        self.step_counter = 0

    def step(self, action):
        # Increment step counter
        self.step_counter += 1

        # Get current mouse position
        x, y = pyautogui.position()
        
        if action == 0:  # Move left
            x = max(0, x - 100)
            pyautogui.moveTo(x, y)
            print(f"Mouse moved left - New Position: {x, y}")
        elif action == 1:  # Move right
            x = min(pyautogui.size()[0], x + 100)
            pyautogui.moveTo(x, y)
            print(f"Mouse moved right - New Position: {x, y}")
        elif action == 2:  # Move up
            y = max(0, y - 100)
            pyautogui.moveTo(x, y)
            print(f"Mouse moved up - New Position: {x, y}")
        elif action == 3:  # Move down
            y = min(pyautogui.size()[1], y + 100)
            pyautogui.moveTo(x, y)
            print(f"Mouse moved down - New Position: {x, y}")
        elif action == 4:  # Click
            pyautogui.click(x, y)
            print(f"Mouse left clicked")
        elif action == 5:  # Type something
            pyautogui.write("Hello")
            print("Typed 'Hello'")

        # Dummy reward (to be improved later)
        reward = -1  # Simple negative reward to encourage shortHelloHelloer episodes

        # End condition (for now, let's just limit the steps)
        done = False

        # Observation: Updated mouse position
        obs = np.array([x, y])

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        return None,None

    def render(self):
        pass  # No need to render anything for now

    def close(self):
        pass
