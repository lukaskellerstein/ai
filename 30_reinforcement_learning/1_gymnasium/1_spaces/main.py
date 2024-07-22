import numpy as np
from gymnasium.spaces import (
    Box,
    Discrete,
    Tuple,
    Dict,
    MultiBinary,
    MultiDiscrete,
    Text,
    Sequence,
    Graph,
)

# ---------------------------------------------------
# Fundamental Spaces
# ---------------------------------------------------

print("-------- Box --------")
box_space = Box(low=-1.0, high=1.0, shape=(3, 2))
print(box_space)
print(box_space.sample())
print(box_space.sample())

print("-------- Discrete --------")
discrete_space = Discrete(5)
print(discrete_space)
print(discrete_space.sample())
print(discrete_space.sample())

print("-------- MultiBinary --------")
observation_space = MultiBinary(5, seed=42)
print(observation_space)
print(observation_space.sample())
print(observation_space.sample())

print("-------- MultiDiscrete --------")
observation_space = MultiDiscrete(np.array([[1, 2, 3], [4, 5, 6]]), seed=42)
print(observation_space)
print(observation_space.sample())
print(observation_space.sample())

print("-------- Text --------")
text_space = Text(min_length=3, max_length=5)
print(text_space)
print(text_space.sample())
print(text_space.sample())


# ---------------------------------------------------
# Composite Spaces
# ---------------------------------------------------

print("-------- Dict --------")
dict_space = Dict(
    {
        "position": Box(low=0, high=1, shape=(2,)),
        "velocity": Box(low=-1, high=1, shape=(2,)),
    }
)
print(dict_space)
print(dict_space.sample())
print(dict_space.sample())

print("-------- Tuple --------")
tuple_space = Tuple((Discrete(2), Box(low=0, high=1, shape=(2,))))
print(tuple_space)
print(tuple_space.sample())
print(tuple_space.sample())


print("-------- Sequence --------")
observation_space = Sequence(Box(0, 1), seed=2)
print(observation_space)
print(observation_space.sample())
print(observation_space.sample())

print("-------- Graph --------")
observation_space = Graph(
    node_space=Box(low=-100, high=100, shape=(3,)), edge_space=Discrete(3), seed=42
)
print(observation_space)
print(observation_space.sample())
print(observation_space.sample())
