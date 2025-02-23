import numpy as np
import gymnasium as gym
import setup
import time
import random  # ✅ Needed for adding randomness in action selection
from collections import deque  # ✅ Keeps track of past states to detect loops

# Load trained Q-table
q_table = np.load("q_table_pygame.npy")

print("\n=== Q-Table (State-Action Values) ===")
print("Dimensions: (Grid Cells = 32x24, Actions = 4)")
print("Each cell stores the value of taking an action (Up=0, Down=1, Left=2, Right=3) in that state.\n")

# Print the Q-table (rounded to 2 decimal places)
for row in range(q_table.shape[0]):
    for col in range(q_table.shape[1]):
        rounded_values = np.round(q_table[row, col, :], 2)
        print(f"State ({row},{col}): {rounded_values}")

# Create environment
env = gym.make("PygameGridEnv-v0")
env.unwrapped.clock.tick(5)  # ✅ Slow down Pygame updates (5 FPS)

state_dict, info = env.reset()
x, y = state_dict["current_position"]
state = (x // env.player_size, y // env.player_size)  # Convert to grid index
done = False

step_number = 0  # ✅ Track the number of steps
epsilon = 0.2  # ✅ Increased exploration chance
loop_detect = deque(maxlen=10)  # ✅ Stores last 10 states to detect loops

print("\n=== Testing the Q-Learning Agent ===")
print(f"Initial State: {state}\n")

while not done:
    step_number += 1  # ✅ Increase step number

    # Check if we are stuck in a loop (same state appears frequently)
    if loop_detect.count(state) > 3:  # ✅ If the same state appears more than 3 times recently, force exploration
        print(f"🔄 Loop detected! Forcing random action at Step {step_number}")
        action = env.action_space.sample()
    else:
        # Select the best action based on Q-table, but allow randomness if Q-values are too close
        max_q_value = np.max(q_table[state[0], state[1], :])
        best_actions = np.where(q_table[state[0], state[1], :] == max_q_value)[0]  # Get all best actions

        # If Q-values are too similar, increase exploration
        if np.ptp(q_table[state[0], state[1], :]) < 0.01:  # ✅ Check if Q-values are almost the same
            action = env.action_space.sample()  # Force random action
        else:
            action = np.random.choice(best_actions) if np.random.rand() > epsilon else env.action_space.sample()

    # Print Q-values for the current state
    print(f"\nStep {step_number} - State ({state[0]}, {state[1]}) Q-values:")
    print(f"  Up (0): {q_table[state[0], state[1], 0]:.2f} | Down (1): {q_table[state[0], state[1], 1]:.2f}")
    print(f"  Left (2): {q_table[state[0], state[1], 2]:.2f} | Right (3): {q_table[state[0], state[1], 3]:.2f}")

    # Show the chosen action
    action_text = ["Up (0)", "Down (1)", "Left (2)", "Right (3)"][action]
    print(f"  Chosen Action: {action_text}")

    # Take action in the environment
    state_dict, reward, terminated, truncated, info = env.step(action)
    x, y = state_dict["current_position"]
    state = (x // env.player_size, y // env.player_size)  # Convert to grid index

    # ✅ Add state to loop detection
    loop_detect.append(state)

    # ✅ Add a delay to slow down the game
    time.sleep(0.3)  # 0.3 seconds delay between moves

    # Stop if the episode ends
    done = terminated or truncated

env.close()
print("\n=== Testing Complete ===")