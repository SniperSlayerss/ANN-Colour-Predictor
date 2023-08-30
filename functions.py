import os
import numpy as np
from PIL import Image
import skimage.transform

# 25k images to iterate
dataset_path = 'Images'
# List all the image files in the directory
image_files = os.listdir(dataset_path)

def resize_64(Path):
    dataset_path = Path
    # List all the image files in the directory
    image_files = os.listdir(dataset_path)

    training_images = []
    
    for i, image in enumerate(image_files):
        image_path = os.path.join(dataset_path, image)
        image = skimage.io.imread(image_path)
        image = skimage.transform.resize(image, (64, 64), order = 0, )
        image_arr = np.array(image)

        training_images.append(image_arr)

        if i % 100 == 0:
            print(i)

    return training_images


def get_dimensions():
    sizes = {}

    for i, image in enumerate(image_files):
        image_path = os.path.join(dataset_path, image)
        image = Image.open(image_path)
        image_arr = np.array(image)
    
        image_size = image_arr.shape
    
        match len(image_size):
            case 3:
                width, height, _ = image_arr.shape
            case 2:
                width, height = image_arr.shape

        if (width, height) not in sizes:
            sizes[(width, height)] = 1
        else:
            sizes[(width, height)] = sizes[(width, height)] + 1