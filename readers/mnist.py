#!/usr/bin/env python3

import numpy as np
import os.path
import pickle
import struct
from scipy.ndimage.filters import gaussian_filter

def read_data():
    mnist_dog_file = 'data/mnist/mnist_dog.pickle'

    if os.path.isfile(mnist_dog_file):
        print('Reading {}...'.format(mnist_dog_file))
        with open(mnist_dog_file, 'rb') as f:
            train_set, test_set = pickle.load(f)
    else:
        print('Cannot find ' + mnist_dog_file)

        print('Reading original MNIST...')
        labels = []
        for filename in ('train-labels-idx1-ubyte', 't10k-labels-idx1-ubyte'):
            with open('data/mnist/' + filename, 'rb') as f:
                magic, num = struct.unpack(">II", f.read(8))
                labels.append(np.fromfile(f, dtype=np.int8))

        images = []
        for filename in ('train-images-idx3-ubyte', 't10k-images-idx3-ubyte'):
            with open('data/mnist/' + filename, 'rb') as f:
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                images.append(np.fromfile(f, dtype=np.uint8).reshape(num, rows*cols))

        print('Preprocessing with DoG filter...')
        images2 = [np.empty((60000, 784 * 2), dtype=np.uint8), np.empty((10000, 784 * 2), dtype=np.uint8)]
        for i in range(2):
            for j in range(labels[i].size):
                im1 = gaussian_filter(images[i][j].reshape((28, 28)), sigma=1)
                im2 = gaussian_filter(images[i][j].reshape((28, 28)), sigma=2)

                im_on = im1.astype(np.int32) - im2.astype(np.int32)
                im_on[im_on < 0] = 0
                im_on = im_on.astype(np.uint8).reshape((784,))

                im_off = im2.astype(np.int32) - im1.astype(np.int32)
                im_off[im_off < 0] = 0
                im_off = im_off.astype(np.uint8).reshape((784,))

                # import pdb
                # pdb.set_trace()

                images2[i][j] = np.append(im_off, im_on)


        train_set = (images2[0], labels[0])
        test_set = (images2[1], labels[1])

        with open(mnist_dog_file, 'wb') as f:
            pickle.dump((train_set, test_set), f)

    return train_set, test_set
