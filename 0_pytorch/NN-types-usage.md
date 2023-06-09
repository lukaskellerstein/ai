# ANN - Artificial neural network

Basic and general NN.

## Perceptron - Single level

Basic NN. Just for educational purposes, no real usage.

## MLP - Multi Level Perceptron

Usage:

> MLPs are typically used for kind of data that has a **tabular-like** topology.

- Text Classification
- Regression Problems
- Anomaly Detection

# CNN - Convolutional neural network

Usage:

> CNNs are typically used for tasks that involve **image or video** data, or any kind of data that has a **grid-like** topology.

- Image Classification: This is probably the most well-known application of CNNs. Given an input image, a CNN can be trained to output a class label for that image. For example, a CNN might be trained to classify images as cats, dogs, or other animals.

- Object Detection: While image classification assigns a single label to an entire image, object detection aims to identify multiple objects within an image and provide a bounding box around each detected object. CNNs play a key role in these systems, often being used to classify each proposed bounding box.

- Semantic Segmentation: In semantic segmentation, the goal is to assign a class label to each pixel in the image. For example, in a street scene, a semantic segmentation model might label each pixel as belonging to a car, a person, the road, a building, etc. CNNs form the backbone of most semantic segmentation models.

- Image Generation: CNNs can also be used in generative models, such as Generative Adversarial Networks (GANs), which can generate new images that resemble a given dataset. These have been used to create realistic-looking, but entirely artificial, images of faces, animals, and many other things.

- Face Recognition: CNNs are often used in face recognition systems, where the goal is to identify or verify the identity of a person based on a digital image or a video frame from a video source.

- Medical Imaging: CNNs have been used extensively in medical imaging for tasks such as detecting tumors in MRI scans, identifying signs of disease in X-rays, and segmenting different types of tissue in histology images.

- Video Analysis: By extending CNNs to operate on video data, they can be used for action recognition, person tracking, and many other video analysis tasks.

# RNN - Recurent Neural Network

Usage:

> Designed to recognize patterns in **sequences** of data, such as text, genomes, handwriting, or the spoken word.

- Natural Language Processing (NLP)
- Speech Recognition
- Time Series Prediction
- Video Analysis
- Music Composition

## LSTM - Long Short-Term Memory

Usage:

- Anomaly Detection in Time Series

## GRU - Gated Recurrent Unit

> Same as LSTM, but less expensive to run.

# Transformer

Usage:

> Used recently for a lot of tasks

- Machine Translation
- Text Summarization
- Sentiment Analysis
- Question Answering
- Text Generation
- Speech Recognition

## ViT - Vision Transformer

- Image Recognition
