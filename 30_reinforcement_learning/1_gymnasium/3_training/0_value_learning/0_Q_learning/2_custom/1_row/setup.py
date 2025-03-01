from gymnasium.envs.registration import register

from env import CustomEnv

register(
    id="CustomEnv-v0",
    entry_point=CustomEnv,  # If defined in the main script
)
