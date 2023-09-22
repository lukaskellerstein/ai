import os
from dotenv import load_dotenv, find_dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command
from azure.ai.ml import Input
from azure.ai.ml.entities import Model
from azure.ai.ml.entities import Data
from azure.ai.ml import Output
from azure.ai.ml.constants import AssetTypes

_ = load_dotenv(find_dotenv())  # read local .env file


# -------------------------------------------
# Variables
# -------------------------------------------
workspace = os.getenv("WORSPACE_NAME")
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RG")

curated_env_name = "responsibleai-ubuntu20.04-py38-cpu@latest"
compute_target = "test-cpu-compute"


# -------------------------------------------
# Client
# -------------------------------------------
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)
print(ml_client)


# -------------------------------------------
# 1. Command (Get data job)
# -------------------------------------------
inputs = {
    "cifar_zip": Input(
        type=AssetTypes.URI_FILE,
        path="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    ),
}

outputs = {
    "cifar": Output(
        type=AssetTypes.URI_FOLDER,
        path=f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace}/datastores/workspaceblobstore/paths/CIFAR-10",
    )
}
print(
    f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace}/datastores/workspaceblobstore/paths/CIFAR-10"
)

job = command(
    code="./model",  # local path where the code is stored
    command="python read_write_data.py --input_data ${{inputs.cifar_zip}} --output_folder ${{outputs.cifar}}",
    inputs=inputs,
    outputs=outputs,
    compute=compute_target,
    environment=curated_env_name,
    experiment_name="get-data-model-1",
    display_name="get-data-job-1",
)
print(job)

# submit the command
returned_job = ml_client.jobs.create_or_update(job)
# get a URL for the status of the job
print(returned_job.studio_url)

ml_client.jobs.stream(returned_job.name)

print(returned_job.name)
print(returned_job.experiment_name)
print(returned_job.outputs.cifar)
print(returned_job.outputs.cifar.path)

# # -------------------------------------------
# # 2. Command (Training job)
# # -------------------------------------------

# inputs = {
#     "cifar": Input(
#         type=AssetTypes.URI_FOLDER,
#         path=f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace}/datastores/workspaceblobstore/paths/CIFAR-10",
#     ),
#     "epoch": 10,
#     "batchsize": 64,
#     "workers": 2,
#     "lr": 0.01,
#     "momen": 0.9,
#     "prtfreq": 200,
#     "output": "./outputs",
# }


# training_job = command(
#     code="./model",  # local path where the code is stored
#     command="python main.py --data-dir ${{inputs.cifar}} --epochs ${{inputs.epoch}} --batch-size ${{inputs.batchsize}} --workers ${{inputs.workers}} --learning-rate ${{inputs.lr}} --momentum ${{inputs.momen}} --print-freq ${{inputs.prtfreq}} --model-dir ${{inputs.output}}",
#     inputs=inputs,
#     compute=compute_target,
#     environment=curated_env_name,
#     experiment_name="testing-model-1",
#     display_name="training-job-1",
# )

# ml_client.jobs.create_or_update(training_job)

# print("------------Training Job----------------")
# print(training_job)
