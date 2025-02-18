import deeplake
from PIL import Image
import numpy as np
import os

ds = deeplake.empty("./animals_deeplake")  # Create the dataset locally


# Find the class_names and list of files that need to be uploaded
dataset_folder = "./animals"

# Find the subfolders, but filter additional files like DS_Store that are added on Mac machines.
class_names = [
    item
    for item in os.listdir(dataset_folder)
    if os.path.isdir(os.path.join(dataset_folder, item))
]

files_list = []
for dirpath, dirnames, filenames in os.walk(dataset_folder):
    for filename in filenames:
        files_list.append(os.path.join(dirpath, filename))


with ds:
    # Create the tensors with names of your choice.
    ds.create_tensor("images", htype="image", sample_compression="jpeg")
    ds.create_tensor("labels", htype="class_label", class_names=class_names)

    # Add arbitrary metadata - Optional
    ds.info.update(description="My first Deep Lake dataset")
    ds.images.info.update(camera_type="SLR")


with ds:
    # Iterate through the files and append to Deep Lake dataset
    for file in files_list:
        label_text = os.path.basename(os.path.dirname(file))
        label_num = class_names.index(label_text)

        # Append data to the tensors
        ds.append({"images": deeplake.read(file), "labels": np.uint32(label_num)})


print(ds.summary())
