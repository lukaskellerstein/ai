import os
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

# -------------------------------------------
# Variables
# -------------------------------------------

workspace = os.getenv("WORSPACE_NAME")
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RG")

# endpoint_name = "my-endpoint-" + datetime.datetime.now().strftime("%m%d%H%M%f")
endpoint_name = "my-endpoint-1111"
model_name = "my-model-1111"

# -------------------------------------------
# -------------------------------------------


# -------------------------------------------
# Client
# -------------------------------------------
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)
print(ml_client)


# -------------------------------------------
# Endpoints object
# -------------------------------------------
# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name, description="this is a sample endpoint", auth_mode="key"
)
print(endpoint)


# -------------------------------------------
# Model object
# -------------------------------------------
model = Model(
    name=model_name,
    path="./model/sklearn_regression_model.pkl",
    description="Model created from run.",
)
ml_client.models.create_or_update(model)
print(model)


# -------------------------------------------
# Environment object
# -------------------------------------------
env = Environment(
    conda_file="./environment/env.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)
print(env)


# -------------------------------------------
# Deployment object
# -------------------------------------------
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code="./environment", scoring_script="score.py"
    ),
    instance_type="Standard_D2as_v4",
    instance_count=1,
)
print(blue_deployment)


# -------------------------------------------
# -------------------------------------------
# A) Deploy to Azure
# -------------------------------------------
# -------------------------------------------

# create endpoint
# ml_client.online_endpoints.begin_create_or_update(endpoint)
# create deployment
ml_client.online_deployments.begin_create_or_update(blue_deployment)
# # create endpoint
# # note: blue deployment takes 100 traffic
# endpoint.traffic = {"blue": 100}
# ml_client.online_endpoints.begin_create_or_update(endpoint)
