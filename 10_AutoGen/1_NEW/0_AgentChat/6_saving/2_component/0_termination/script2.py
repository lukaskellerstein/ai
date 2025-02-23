import joblib
from autogen_agentchat.conditions import MaxMessageTermination, StopMessageTermination

# Deserialize from file
or_term_config = joblib.load("or_term_config.joblib")

# Load into component
or_termination = MaxMessageTermination(5) | StopMessageTermination()
new_or_termination = or_termination.load_component(or_term_config)

print(new_or_termination)

