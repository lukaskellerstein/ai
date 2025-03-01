from gymnasium.envs.registration import register

from env import Grid2DEnv

register(
    id="Grid2DEnv-v0",
    entry_point=Grid2DEnv,  # If defined in the main script
)
