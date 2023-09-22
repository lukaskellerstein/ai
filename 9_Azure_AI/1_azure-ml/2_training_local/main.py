import os
import logging
import datetime
from dotenv import load_dotenv, find_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

_ = load_dotenv(find_dotenv())  # read local .env file

workspace = os.getenv("WORSPACE_NAME")
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RG")

# -------------------------------------------
# Client
# -------------------------------------------
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

print(ml_client)


# -------------------------------------------
# Environment (Custom)
# -------------------------------------------
custom_env_name = "test-model-1-training-env"

custom_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for Training test-model-1 job",
    tags={"env-type": "test"},
    env_file=os.path.join("environment", "env.yaml"),
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)
custom_job_env = ml_client.environments.create_or_update(custom_job_env)

print(custom_job_env)

# -------------------------------------------
# FINISH !!!!!!!!!!!!!!!!!!!
# -------------------------------------------
