import gymnasium as gym
import setup

# Create an instance of the custom environment
env = gym.make("CustomEnv-v0")

# Interact with the environment
state, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Sample a random action
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated

env.close()
