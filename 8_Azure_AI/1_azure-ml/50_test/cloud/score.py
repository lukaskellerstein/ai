import os
import logging
import json
import numpy
import torch
from model import MLP
import torch.nn as nn


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "saved_weights.pt")

    print("model_path: ", model_path)

    # Hyper-parameters
    input_size = 300  # this should match the length of your vectorized text
    hidden_size = 200
    output_size = 1
    model = MLP(input_size, hidden_size, output_size)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model 1: request received")
    data = json.loads(raw_data)["data"]

    print(data)

    input_data = torch.tensor(data)

    # get prediction
    with torch.no_grad():
        output = model(input_data)

        print("output: ", output)

        # classes = ["chicken", "turkey"]
        # softmax = nn.Softmax(dim=1)
        # pred_probs = softmax(output).numpy()[0]
        # index = torch.argmax(output, 1)

    result = {"result": output.detach()}
    return result
