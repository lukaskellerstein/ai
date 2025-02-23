import numpy as np
import gymnasium as gym
import setup

# Load trained Q-table
q_table = np.load("q_table.npy")
print("\n=== Q-Table (State-Action Values) ===")
print("Rows = States (Agent's Position), Columns = Actions (0=Left, 1=Right)\n")
print(np.round(q_table, 2))

# Create environment
env = gym.make("CustomEnv-v0")
state, info = env.reset()
done = False

print("\n=== Testing the Q-Learning Agent ===")
print(f"Initial State: {state}")

while not done:
    # Select the best action based on Q-table
    action = np.argmax(q_table[state, :])

    # Print Q-values for the current state
    print("\nCurrent Q-values for state", state)
    print(f"  Left (0): {q_table[state, 0]:.3f} | Right (1): {q_table[state, 1]:.3f}")
    
    # Show the chosen action
    action_text = "Left (0)" if action == 0 else "Right (1)"
    print(f"  Chosen Action: {action_text}")

    # Take the action
    state, reward, terminated, truncated, info = env.step(action)

    # Render environment (visual representation)
    env.render()

    # Stop if the episode ends
    done = terminated or truncated

env.close()
print("\n=== Testing Complete ===")