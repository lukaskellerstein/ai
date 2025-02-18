import os
from dotenv import load_dotenv, find_dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command
from azure.ai.ml import Input
from azure.ai.ml.entities import Model

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
# Environment (Existing in cloud)
# -------------------------------------------
curated_env_name = "AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu@latest"
# curated_env_name = "responsibleai-ubuntu20.04-py38-cpu@latest"

# -------------------------------------------
# Compute
# -------------------------------------------
# gpu_compute_target = "ai-paid-compute"
gpu_compute_target = "new-ai-compute"

# -------------------------------------------
# Command (Training job)
# -------------------------------------------
training_job = command(
    inputs=dict(
        num_epochs=30, learning_rate=0.001, momentum=0.9, output_dir="./outputs"
    ),
    compute=gpu_compute_target,
    environment=curated_env_name,
    code="./model/",  # location of source code
    command="python main.py --num_epochs ${{inputs.num_epochs}} --output_dir ${{inputs.output_dir}}",
    experiment_name="testing-model-1",
    display_name="my-new-testing-job",
)

training_job = ml_client.jobs.create_or_update(training_job)

print("------------Training Job----------------")
print(training_job)

# -------------------------------------------
# Model - Save
# -------------------------------------------
# if training_job.status == "Completed":

#     # First let us get the run which gave us the best result
#     best_run = training_job.properties["best_child_run_id"]

#     # lets get the model from this run
#     model = Model(
#         # the script stores the model as "outputs"
#         path="azureml://jobs/{}/outputs/artifacts/paths/outputs/".format(best_run),
#         name="run-model-example",
#         description="Model created from run.",
#         type="custom_model",
#     )

# else:
#     print(
#         "Sweep job status: {}. Please wait until it completes".format(
#             returned_sweep_job.status
#         )
#     )
