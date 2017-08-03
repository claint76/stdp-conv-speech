#!/usr/bin/env python3

import os
import numpy as np
from scipy.ndimage import imread

def read_data():
    labels = []

    faces_images = []
    dirpath = 'data/caltech101/101_ObjectCategories_resized_DoG/Faces'
    for filename in os.listdir(dirpath):
        faces_images.append(imread(os.path.join(dirpath, filename)).flatten())

    motors_images = []
    dirpath = 'data/caltech101/101_ObjectCategories_resized_DoG/Motorbikes'
    for filename in os.listdir(dirpath):
        motors_images.append(imread(os.path.join(dirpath, filename)).flatten())

    train_images = faces_images[:200] + motors_images[:200]
    train_labels = [0] * 200 + [1] * 200

    test_images = faces_images[200:] + motors_images[200:]
    test_labels = [0] * len(faces_images[200:]) + [1] * len(motors_images[200:])

    train_images = np.asarray(train_images)
    test_images = np.asarray(test_images)
    train_labels = np.asarray(train_labels, dtype=np.int8)
    test_labels = np.asarray(test_labels, dtype=np.int8)

    return ((train_images, train_labels), (test_images, test_labels))
