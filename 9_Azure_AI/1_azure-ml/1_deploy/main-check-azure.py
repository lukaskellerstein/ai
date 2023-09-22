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
# -------------------------------------------


# -------------------------------------------
# Client
# -------------------------------------------
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)
print(ml_client)


# -------------------------------------------
# -------------------------------------------
# A) Check in Azure
# -------------------------------------------
# -------------------------------------------

cloud_endpoint = ml_client.online_endpoints.get(name=endpoint_name)
print("Cloud Endpoint: ", cloud_endpoint)

print("Kind\tLocation\tName")
print("-------\t----------\t------------------------")
for endpoint in ml_client.online_endpoints.list():
    print(f"{endpoint.kind}\t{endpoint.location}\t{endpoint.name}")


# -------------------------------------------
# Get Logs Azure
# -------------------------------------------
logs = ml_client.online_deployments.get_logs(
    name="blue", endpoint_name=endpoint_name, lines=50
)
print("Logs: ", logs)
