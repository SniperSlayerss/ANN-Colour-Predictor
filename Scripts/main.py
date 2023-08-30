import ArtificialNeuralNetwork
import pickle
import skimage.transform
import numpy as np
import os
import torch

neural_net = ArtificialNeuralNetwork.ArtificialNeuralNetwork(1)
 
lables = None
with open('labels1.pickle', 'rb') as handle:
    labels = pickle.load(handle)

with open('resized_images.pickle', 'rb') as handle:
    images = pickle.load(handle)

training_data = [image.reshape(-1) for image in images]
print(np.array(training_data).shape)
print(labels.shape)

if (labels is not None) and (training_data is not None):
    neural_net.train(np.array(training_data), labels, (12288, 100, 100, 10), 10000, 0.01, False, True)