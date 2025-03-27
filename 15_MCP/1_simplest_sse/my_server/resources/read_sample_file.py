import os

RESOURCES_DIR = os.path.dirname(__file__)

def read_sample_file():

    # print(os.getcwd())

    # print(os.path.dirname(__file__))

    RESOURCES_DIR = os.path.dirname(__file__)

    isExist = os.path.exists(RESOURCES_DIR)
    # print("RESOURCES_DIR exists: ", isExist)
    isExist = os.path.exists(os.path.join(RESOURCES_DIR, "sample.txt"))
    # print("sample.txt exists: ", isExist)

    with open(os.path.join(RESOURCES_DIR, "sample.txt"), 'r') as file:
        content = file.read()

    # print("Content")
    # print(content)

    return content