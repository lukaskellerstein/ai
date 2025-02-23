from autogen_agentchat.conditions import MaxMessageTermination, StopMessageTermination
import joblib

max_termination = MaxMessageTermination(5)
stop_termination = StopMessageTermination()

or_termination = max_termination | stop_termination

# Save the component
or_term_config = or_termination.dump_component()
print("Config: ", or_term_config.model_dump_json())

# Serialize the component
joblib.dump(or_term_config, "or_term_config.joblib")
