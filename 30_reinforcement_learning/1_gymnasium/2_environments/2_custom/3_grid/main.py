import gymnasium as gym
import setup

gym.pprint_registry()

# Create an instance of the custom environment
env = gym.make("Grid2DEnv-v0")

# Interact with the environment
state, info = env.reset()
done = False

current_step = 0
while not done:
    print(f"Step: {current_step}")
    action = env.action_space.sample()  # Sample a random action
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
    current_step += 1

env.close()
