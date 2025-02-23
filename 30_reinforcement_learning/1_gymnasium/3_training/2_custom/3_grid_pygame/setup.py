from gymnasium.envs.registration import register

from env import PygameGridEnv

register(
    id="PygameGridEnv-v0",
    entry_point=PygameGridEnv,  
)
