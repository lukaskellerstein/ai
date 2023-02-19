import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ------------------------------------------------------------
# SOURCE: https://www.geeksforgeeks.org/single-layer-perceptron-in-tensorflow/
# ------------------------------------------------------------

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

len(x_train)
len(x_test)
x_train[0].shape
plt.matshow(x_train[0])

# Normalizing the dataset
x_train = x_train / 255
x_test = x_test / 255

# Flatting the dataset in order
# to compute for model building
x_train_flatten = x_train.reshape(len(x_train), 28 * 28)
x_test_flatten = x_test.reshape(len(x_test), 28 * 28)


# -------------------
# Building the model and adding layers
# -------------------

model = keras.Sequential(
    [keras.layers.Dense(10, input_shape=(784,), activation="sigmoid")]
)

# -------------------
# Training the model
# -------------------

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Training the model on the training set
model.fit(x_train_flatten, y_train, epochs=5)


# ????????
model.evaluate(x_test_flatten, y_test)


# -------------------
# Predicting ...
# -------------------
# ????????
