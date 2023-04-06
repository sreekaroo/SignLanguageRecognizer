from PIL import Image
import itertools
import random
import string
from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import random as rnd
import string
from pandas import DataFrame
from math import log2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def dataframe_to_array(dataframe):
    """ converts rows of df into np arrays. Returns feature rows, and labels
    """
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1.iloc[:, 1:].to_numpy()
    targets_array = dataframe1['label'].to_numpy()
    return inputs_array, targets_array


# rendering pixels in row
def row_to_img_grid(row):
    sqrt = int(np.sqrt(len(row)))
    return np.reshape(row, (sqrt, sqrt))


def render_row(row):
    plt.imshow(row_to_img_grid(row), cmap="gray")
    plt.show()


def do_nb(xtrain, ytrain, xtest, ytest):
    clf = MultinomialNB()
    clf.fit(xtrain, ytrain)
    score = clf.score(xtest, ytest)

    return score


def crop_pixel_row(row, crop_size=3):
    img_grid = row_to_img_grid(row)
    img_grid = img_grid[crop_size:len(img_grid) - crop_size, crop_size:len(img_grid) - crop_size]
    rowLen = (len(img_grid)) ** 2
    return np.reshape(img_grid, rowLen)


# trying to crop image uniformly to increase accurcay
def apply_crop(features, crop_size=3):
    new_features = np.array([crop_pixel_row(row, crop_size=crop_size) for row in features])
    return new_features


def random_crop(image, crop_height, crop_width):
    """Perform random crop augmentation on a 2D numpy array image."""

    # Get image shape and desired crop dimensions
    image_height, image_width = image.shape

    # Generate random coordinates for the top-left corner of the crop
    x = rnd.randint(0, image_width - crop_width)
    y = rnd.randint(0, image_height - crop_height)

    # Extract the crop from the image
    cropped_image = image[y:y + crop_height, x:x + crop_width]

    # converting to pillow image object and resizing back to original size
    imgObj = Image.fromarray(cropped_image.astype(np.uint8))
    imgObj = imgObj.resize((image_height, image_width))
    imgObj = np.array(imgObj)
    return imgObj


def apply_random_image_crop(X, crop_height, crop_width):
    def row_crop(row):
        img_grid = row_to_img_grid(row)
        img_grid = random_crop(img_grid, crop_height, crop_width)
        rowLen = (img_grid.shape[0]) ** 2
        return np.reshape(img_grid, rowLen)

    return np.array([row_crop(vec) for vec in X])
