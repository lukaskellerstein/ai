import numpy as np
import gymnasium as gym
import setup
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DQN

# Create environment
env = gym.make("Grid2DEnv-v0")

# Create a vectorized environment for SB3 algorithms
vec_env = make_vec_env("Grid2DEnv-v0", n_envs=1)

# Train DQN
dqn_model = DQN("MlpPolicy", vec_env, verbose=1)
dqn_model.learn(total_timesteps=1000, log_interval=10)
dqn_model.save("dqn_grid")