#!/usr/bin/env python3

import numpy as np
import os.path
import pickle
import struct
from scipy.ndimage import correlate

def read_data():
    mnist_dog_file = 'data/mnist/mnist_dog_2.pickle'

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
        images2 = [np.empty((60000, 784 * 2)), np.empty((10000, 784 * 2))]

        sz = 7
        sigma1 = 1
        sigma2 = 2
        x = np.tile(np.arange(1, sz+1), (7, 1))
        y = x.transpose()
        d2 = np.square(x-sz/2-.5) + np.square(y-sz/2-.5)
        filter = 1/np.sqrt(2*np.pi) * ( 1/sigma1 * np.exp(-d2/2/(sigma1**2)) - 1/sigma2 * np.exp(-d2/2/(sigma2**2)) )
        filter = filter - filter.mean()
        filter = filter / filter.max()

        for i in range(2):
            for j in range(labels[i].size):
                im = correlate(images[i][j].astype(np.float64).reshape((28, 28)), filter)

                im_on = im.copy()
                im_on[im_on < 0] = 0
                im_on = im_on.reshape((784,))

                im_off = -im.copy()
                im_off[im_off < 0] = 0
                im_off = im_off.reshape((784,))

                images2[i][j] = np.append(im_off, im_on)

        train_set = (images2[0], labels[0])
        test_set = (images2[1], labels[1])

        with open(mnist_dog_file, 'wb') as f:
            pickle.dump((train_set, test_set), f)

    return train_set, test_set
