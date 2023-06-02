import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ------------------------------------------------------------
#
# Image classification with a single layer perceptron
#
# Data are images of handwritten digits (0-9)
#
# SOURCE: https://www.youtube.com/watch?v=O5xeyoRL95U&t=1580s
# ------------------------------------------------------------

# -------------------
# 1. Get data
# -------------------
(train_images, train_labels), (
    test_images,
    test_labels,
) = keras.datasets.mnist.load_data()

# -------------------
# 2. Building the model and adding layers
# -------------------

model = keras.Sequential(
    [
        # input layer ------------------
        keras.layers.Flatten(input_shape=(28, 28)),
        # hidden layers -----------------
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        # output layer ------------------
        keras.layers.Dense(10, activation="softmax"),
    ]
)

# -------------------
# 3. Training the model
# -------------------

# OPTIMIZERS
# Adadelta: Optimizer that implements the Adadelta algorithm.
# Adagrad: Optimizer that implements the Adagrad algorithm.
# Adam: Optimizer that implements the Adam algorithm.
# Adamax: Optimizer that implements the Adamax algorithm.
# Ftrl: Optimizer that implements the FTRL algorithm.
# Nadam: Optimizer that implements the Nadam algorithm.
# Optimizer: Abstract optimizer base class.
# RMSprop: Optimizer that implements the RMSprop algorithm.
# SGD: Gradient descent (with momentum) optimizer.

# LOSS
# KLD: Computes Kullback-Leibler divergence loss between y_true & y_pred.
# MAE: Computes the mean absolute error between labels and predictions.
# MAPE: Computes the mean absolute percentage error between y_true & y_pred.
# MSE: Computes the mean squared error between labels and predictions.
# MSLE: Computes the mean squared logarithmic error between y_true & y_pred.
# binary_crossentropy: Computes the binary crossentropy loss.
# binary_focal_crossentropy: Computes the binary focal crossentropy loss.
# categorical_crossentropy: Computes the categorical crossentropy loss.
# categorical_hinge: Computes the categorical hinge loss between y_true & y_pred.
# cosine_similarity: Computes the cosine similarity between labels and predictions.
# deserialize: Deserializes a serialized loss class/function instance.
# get: Retrieves a Keras loss as a function/Loss class instance.
# hinge: Computes the hinge loss between y_true & y_pred.
# huber: Computes Huber loss value.
# kl_divergence: Computes Kullback-Leibler divergence loss between y_true & y_pred.
# kld: Computes Kullback-Leibler divergence loss between y_true & y_pred.
# kullback_leibler_divergence: Computes Kullback-Leibler divergence loss between y_true & y_pred.
# log_cosh: Logarithm of the hyperbolic cosine of the prediction error.
# logcosh: Logarithm of the hyperbolic cosine of the prediction error.
# mae: Computes the mean absolute error between labels and predictions.
# mape: Computes the mean absolute percentage error between y_true & y_pred
# mean_absolute_error: Computes the mean absolute error between labels and predictions.
# mean_absolute_percentage_error: Computes the mean absolute percentage error between y_true & y_pred.
# mean_squared_error: Computes the mean squared error between labels and predictions.
# mean_squared_logarithmic_error: Computes the mean squared logarithmic error between y_true & y_pred.
# mse: Computes the mean squared error between labels and predictions.
# msle: Computes the mean squared logarithmic error between y_true & y_pred.
# poisson: Computes the Poisson loss between y_true and y_pred.
# serialize: Serializes loss function or Loss instance.
# sparse_categorical_crossentropy: Computes the sparse categorical crossentropy loss.
# squared_hinge: Computes the squared hinge loss between y_true & y_pred.

# METRICS
# AUC: Approximates the AUC (Area under the curve) of the ROC or PR curves.
# Accuracy: Calculates how often predictions equal labels.
# BinaryAccuracy: Calculates how often predictions match binary labels.
# BinaryCrossentropy: Computes the crossentropy metric between the labels and predictions.
# BinaryIoU: Computes the Intersection-Over-Union metric for class 0 and/or 1.
# CategoricalAccuracy: Calculates how often predictions match one-hot labels.
# CategoricalCrossentropy: Computes the crossentropy metric between the labels and predictions.
# CategoricalHinge: Computes the categorical hinge metric between y_true and y_pred
# CosineSimilarity: Computes the cosine similarity between the labels and predictions.
# FalseNegatives: Calculates the number of false negatives.
# FalsePositives: Calculates the number of false positives.
# Hinge: Computes the hinge metric between y_true and y_pred.
# IoU: Computes the Intersection-Over-Union metric for specific target classes.
# KLDivergence: Computes Kullback-Leibler divergence metric between y_true and
# LogCoshError: Computes the logarithm of the hyperbolic cosine of the prediction error.
# Mean: Computes the (weighted) mean of the given values.
# MeanAbsoluteError: Computes the mean absolute error between the labels and predictions.
# MeanAbsolutePercentageError: Computes the mean absolute percentage error between y_true and
# MeanIoU: Computes the mean Intersection-Over-Union metric.
# MeanMetricWrapper: Wraps a stateless metric function with the Mean metric.
# MeanRelativeError: Computes the mean relative error by normalizing with the given values.
# MeanSquaredError: Computes the mean squared error between y_true and y_pred.
# MeanSquaredLogarithmicError: Computes the mean squared logarithmic error between y_true and
# MeanTensor: Computes the element-wise (weighted) mean of the given tensors.
# Metric: Encapsulates metric logic and state.
# OneHotIoU: Computes the Intersection-Over-Union metric for one-hot encoded labels.
# OneHotMeanIoU: Computes mean Intersection-Over-Union metric for one-hot encoded labels.
# Poisson: Computes the Poisson metric between y_true and y_pred.
# Precision: Computes the precision of the predictions with respect to the labels.
# PrecisionAtRecall: Computes best precision where recall is >= specified value.
# Recall: Computes the recall of the predictions with respect to the labels.
# RecallAtPrecision: Computes best recall where precision is >= specified value.
# RootMeanSquaredError: Computes root mean squared error metric between y_true and y_pred.
# SensitivityAtSpecificity: Computes best sensitivity where specificity is >= specified value.
# SparseCategoricalAccuracy: Calculates how often predictions match integer labels.
# SparseCategoricalCrossentropy: Computes the crossentropy metric between the labels and predictions.
# SparseTopKCategoricalAccuracy: Computes how often integer targets are in the top K predictions.
# SpecificityAtSensitivity: Computes best specificity where sensitivity is >= specified value.
# SquaredHinge: Computes the squared hinge metric between y_true and y_pred.
# Sum: Computes the (weighted) sum of the given values.
# TopKCategoricalAccuracy: Computes how often targets are in the top K predictions.
# TrueNegatives: Calculates the number of true negatives.
# TruePositives: Calculates the number of true positives.

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Training the model on the training set
model.fit(
    train_images,
    train_labels,
    epochs=5,
    # batch_size=10,
)


# Evaluating accuracy on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_accuracy)

# -------------------
# 4. Predicting ...
# -------------------
predictions = model.predict(test_images)
