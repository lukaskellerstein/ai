import os
from dotenv import load_dotenv, find_dotenv
from azure.ai.ml import MLClient
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
# Client
# -------------------------------------------
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)


# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# Invoke the Endpoint (Test the endpoint)
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# test the blue deployment with some sample data
response = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_name,
    deployment_name="blue",
    request_file="./environment/request.json",
)
print("Response: ", response)
