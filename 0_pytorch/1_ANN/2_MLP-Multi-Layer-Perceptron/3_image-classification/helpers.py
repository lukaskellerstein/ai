from PIL import Image
import io
import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = Image.open(buf).convert("RGB")
    image = np.array(image)
    # Convert RGB to BGR
    image = image[:, :, ::-1].copy()  # Make a copy here to ensure positive strides
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).float() / 255  # CHW and normalize
    return image
