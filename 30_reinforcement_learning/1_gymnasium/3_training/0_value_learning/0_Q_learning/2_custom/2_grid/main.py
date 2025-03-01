import numpy as np
import gymnasium as gym
import setup

# Create environment
env = gym.make("Grid2DEnv-v0")
_, _ = env.reset()

# print(state)
# # ???
# state = state.astype(int)
# print(state)

# Load trained Q-table
q_table = np.load("q_learning_q_table.npy")
print("=========================================")
print("   Q-Table (State-Action Values)")
print("Dimensions: (Rows = 5, Columns = 5, Actions = 4)")
print("Each cell stores the value of taking an action (Up=0, Down=1, Left=2, Right=3) in that state.\n")

for row in range(5):
    for col in range(5):
        rounded_values = np.round(q_table[row, col, :], 2)  # Round values to 2 decimal places
        print(f"State ({row},{col}): {rounded_values}")

print("=========================================\n")
print("=========================================")
print("========== Testing the Agent ============")
# print(f"Initial State: {state}")

# done = False
# while not done:
#     print("----------------------------")
#     print(f"Step {env.unwrapped.step_counter}")
#     print("----------------------------")

#     print(f"State 1a {state}")
#     row, col = map(int, state)  # Get current position
#     print(f"State 1b ({row}, {col}) ")

#     # Select the best action based on Q-table
#     action = np.argmax(q_table[row, col, :])

#     # Print Q-values for the current state
#     # print(f"\nState ({row}, {col}) ")
#     print(f"Q-values:  Up (0): {q_table[row, col, 0]:.2f} | Down (1): {q_table[row, col, 1]:.2f} | Left (2): {q_table[row, col, 2]:.2f} | Right (3): {q_table[row, col, 3]:.2f}")

#     # Show the chosen action
#     action_text = ["Up (0)", "Down (1)", "Left (2)", "Right (3)"][action]
#     print(f"Chosen Action: {action_text}")

#     # Take action
#     state, reward, terminated, truncated, info = env.step(action)

#     # The next state
#     print(f"Next State {state}")
#     state = state.astype(int)
#     print(f"Next State ({row}, {col}) ")

#     # Render environment
#     env.render()

#     # Stop if the episode ends
#     done = terminated or truncated

# env.close()
# print("=========================================")
# print("\n========== Testing Complete ===========")
