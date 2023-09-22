import os
from dotenv import load_dotenv, find_dotenv
from azureml.core import Workspace
from azureml.core.experiment import Experiment

_ = load_dotenv(find_dotenv())  # read local .env file

name = os.getenv("WORSPACE_NAME")
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RG")


# -------------------------------------------
# Workspace
# -------------------------------------------

ws = Workspace.get(
    name=name,
    subscription_id=subscription_id,
    resource_group=resource_group,
)

print(ws)

# -------------------------------------------
# Experiment
# -------------------------------------------

experiment = Experiment(workspace=ws, name="test-experiment")
