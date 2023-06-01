import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

# ------------------------------------------------------------
#
# ???
#
# SOURCE: https://www.udemy.com/course/deeplearning/
# ------------------------------------------------------------

# -------------------
# Get data
# -------------------
# no data needed for this example


# -------------------
# Building the model and adding layers
# -------------------

model = keras.Sequential(
    [
        # hidden layers -----------------
        # Convolution
        keras.layers.Conv2D(
            filters=32, kernel_size=3, activation="relu", input_shape=[64, 64, 3]
        ),
        # Pooling
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        # Adding a second convolutional layer
        keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        # Flattening
        keras.layers.Flatten(),
        # Full Connection
        keras.layers.Dense(units=128, activation="relu"),
        # output layer ------------------
        keras.layers.Dense(units=1, activation="sigmoid"),
    ]
)

# -------------------
# Training the model
# -------------------

# load the weights from pre-trained model
model.load_weights("pretrained_ckpt")

# -------------------
# Predicting ...
# -------------------

test_image = keras.utils.load_img(
    "dataset/single_prediction/cat_or_dog_2.jpg", target_size=(64, 64)
)
test_image = keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"
print(prediction)
