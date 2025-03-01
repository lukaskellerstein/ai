from typing import Any, Literal
import gymnasium as gym
from gymnasium import spaces
import pyautogui
import numpy as np
import time
from omniparser_client import analyze_image
from openai_client import call_with_image

class ComputerControlEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(ComputerControlEnv, self).__init__()

        # Action space: Move mouse in 8 directions and click (9 discrete actions)
        self.action_space = spaces.Discrete(9)
        
        # Action descriptions for logging
        self.action_descriptions = {
            0: "Mouse Move Left",
            1: "Mouse Move Right",
            2: "Mouse Move Up",
            3: "Mouse Move Down",
            4: "Mouse Move Up Left",
            5: "Mouse Move Up Right",
            6: "Mouse Move Down Left",
            7: "Mouse Move Down Right",
            5: "Mouse click"
        }

        print("--------------------------------------------")
        print("ACTION SPACE")
        print(self.action_descriptions)
        print("--------------------------------------------")

        # Observation space: Screen size (normalized coordinates)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        print("----------------------------")
        print("OBSERVATION SPACE")
        print(self.observation_space)
        print("----------------------------")

        self.target_position = None
        self.current_position = None

        # Initialize step counter
        self.step_counter = 0


    # ---------------------------------------------
    # Reset the environment
    # AKA INITIALIZE THE ENVIRONMENT to his default state after each "Learning Episode"
    # ---------------------------------------------
    def reset(self):
        """Resets the environment by taking a screenshot and finding the target."""
        
        print("Waiting 5 seconds to switch to the target window...")
        time.sleep(5)
        print("Taking a screenshot...")

        # Take a screenshot
        screenshot = pyautogui.screenshot()
        print("Screenshot taken.")

        # Analyze the screenshot
        print("Analyzing the screenshot...")
        parsed_data = analyze_image(screenshot)
        print("Screenshot analyzed.")

        # Find a red circle and get his ID

        system_message = """
        You are a computer vision expert. 
        You will receive a screenshot of a computer screen and you have to find the UI element based on the user instructions. 
        Return the ID of the UI element and his coordinates in the format: {'id': <ID>, 'coordinates': [x1, y1, x2, y2]}. 
        Return just this JSON object, nothing else, no text or description is needed.
        Do not format the response in the markdown code block.
        If you cannot find the UI element, return text 'NOT FOUND'."""
        message = f"""Find the red circle in the screenshot. 
        Here are the coordinates of all UI elements on the screen:
        {parsed_data["parsed_content_list"]}
        """

        print("Calling GPT-4o...")
        response = call_with_image(system_message, message, parsed_data["parsed_image_base64"])
        # print("Response from GPT-4o:")
        # print(response)
        # print(type(response))
        # print(response.choices[0])
        # print(type(response.choices[0]))

        response_json = response.choices[0].message.content
        print("Response from GPT-4o:")
        print(response_json)
        print(type(response_json))
        # Parse the response
        if response_json == "NOT FOUND":
            print("UI element not found.")
            self.ui_id = None
            self.ui_target_position = None
        else:
            response_dict = eval(response_json)
            print(response_dict)
            print(type(response_dict))
            # Extract the ID and coordinates
            self.ui_id = response_dict["id"]
            print("UI ID:", self.ui_id)
            self.ui_target_position = response_dict["coordinates"]
            print("UI Target position:", self.ui_target_position)

        # Set the target position
        self.target_position = self.ui_target_position
        # Set current mouse position
        self.current_position = np.array(pyautogui.position()) / np.array(pyautogui.size())

        return self._get_obs(), {}

    def step(self, action) -> tuple[Any, Any, Any, Literal[False], dict]:
        """Executes an action to move the mouse."""
        move_step = 0.05  # Step size in normalized coordinates
        
        # Action mapping
        moves = {
            0: (-move_step, 0),    # Left
            1: (move_step, 0),     # Right
            2: (0, -move_step),    # Up
            3: (0, move_step),     # Down
            4: (-move_step, -move_step),  # Up-Left
            5: (move_step, -move_step),   # Up-Right
            6: (-move_step, move_step),   # Down-Left
            7: (move_step, move_step),    # Down-Right
            8: "click"  # Click action
        }
        
        if action == 8:
            pyautogui.click()
            print("Mouse clicked.")
        else:
            dx, dy = moves[action]
            new_position = np.clip(self.current_position + np.array([dx, dy]), 0, 1)
            screen_size = np.array(pyautogui.size())
            pyautogui.moveTo(*(new_position * screen_size))
            self.current_position = new_position
            print(f"Mouse moved to {self.current_position}")
        
        # Reward function
        distance = np.linalg.norm(self.current_position - self.target_position)
        reward = -distance  # Negative distance to encourage shorter paths
        done = distance < 0.02  # Consider task complete if within range
        
        return self._get_obs(), reward, done, {}

    def render(self):
        pass  # No need to render anything for now

    def close(self):
        pass

    def _get_obs(self):
        """Returns the current observation (normalized coordinates)."""
        return np.concatenate((self.current_position, self.target_position))
    
