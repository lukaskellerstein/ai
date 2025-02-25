import time
from env import ComputerControlEnv

env = ComputerControlEnv()

obs, _ = env.reset()

# AGENT --------------------------------------------
NUMBER_OF_STEPS = 10
for _ in range(NUMBER_OF_STEPS):  

    # Random action
    action = env.action_space.sample()  

    obs, reward, done, _, _ = env.step(action)

    time.sleep(0.5)  # Add delay to see the effects
# --------------------------------------------

env.close()