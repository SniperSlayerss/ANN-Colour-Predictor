"""
Predict average colour of an image based on pre defined r,g,b values
"""
import numpy as np
from PIL import Image
import pickle
import os
from scipy.spatial.distance import cdist
import functions

colour_vals = np.array([
    np.array([255,0,0]),    # red
    np.array([255,127,0]),  # orange
    np.array([255,255,0]),  # yellow
    np.array([127,255,0]),  # green yellow
    np.array([0,255,0]),    # green
    np.array([0,255,127]),  # green cyan
    np.array([0,255,255]),  # cyan
    np.array([0,127,255]),  # blue cyan
    np.array([0,0,255]),    # blue
    np.array([255,0,0]),    # blue magenta
    np.array([127,0,255]),  # magenta
    np.array([255,0,127]),  # red magenta
    np.array([0,0,0]),      # black
    np.array([255,255,255]) # white
])

val_colours = {
    0  : "red",
    1  : "orange",
    2  : "yellow",
    3  : "green yellow",
    4  : "green",
    5  : "green cyan",
    6  : "cyan",
    7  : "blue cyan",
    8  : "blue",
    9  : "blue magenta",
    10 : "magenta",
    11 : "red magenta",
    12 : "black",
    13 : "white",
}

# create labels for dataset of images
def find_average_colour():
    # 25k images to iterate
    dataset_path = 'Images'
    # resize images
    resized_images = functions.resize_64(dataset_path)

    with open('resized_images.pickle', 'wb') as handle:
        pickle.dump(resized_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Create empty lists to store images and labels
    labels_list = []

    # Load and convert images to numpy arrays
    for i, image in enumerate(resized_images):
        image_array = np.array(image)
        image_mean = np.mean(image_array, axis = 1)
        image_mean = np.mean(image_mean, axis = 0)

        closest_colour = None
        closest_distance = np.inf
        for index, col_vector in enumerate(colour_vals):
            distance = np.linalg.norm(col_vector - image_mean)
            
            if distance < closest_distance:
                closest_distance = distance
                closest_colour = index

        # add class
        labels_list.append(closest_colour)
        if i % 100 == 0:
            print(i)
        #print(val_colours[closest_colour])

    # Convert the lists to numpy arrays for further processing
    return np.array(labels_list)

def repeat_colours(colours, height, width):
    # Reshape vectors to enable broadcasting
    colours = colours[:, None, None, :]

    # Repeat the vectors to match the grid dimensions
    repeated_colours = np.repeat(colours, height, axis=1)
    repeated_colours = np.repeat(repeated_colours, width, axis=2)

    return repeated_colours

labels = find_average_colour()

with open('labels1.pickle', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('labels1backup.pickle', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)