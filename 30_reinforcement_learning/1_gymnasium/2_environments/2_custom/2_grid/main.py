import gymnasium as gym
import setup

# Print the registry of environments
# gym.pprint_registry()

# Create an instance of the custom environment
env = gym.make("Grid2DEnv-v0")

# Interact with the environment
state, info = env.reset()
done = False

# AGENT --------------------------------------------
while not done:

    # Agent selects an action = Sample a random action
    action = env.action_space.sample()  
    
    # Agent applies the action
    state, reward, terminated, truncated, info = env.step(action)

    env.render()
    done = terminated or truncated
# --------------------------------------------

env.close()
