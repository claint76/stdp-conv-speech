#!/usr/bin/env python3

import numpy as np
import pickle
import struct

def read_mnist():
    labels = []
    for filename in ('train-labels-idx1-ubyte', 't10k-labels-idx1-ubyte'):
        with open('data/' + filename, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels.append(np.fromfile(f, dtype=np.int8))

    images = []
    for filename in ('train-images-idx3-ubyte-DoG-ON', 't10k-images-idx3-ubyte-DoG-ON', \
            'train-images-idx3-ubyte-DoG-OFF', 't10k-images-idx3-ubyte-DoG-OFF'):
        with open('data/' + filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images.append(np.fromfile(f, dtype=np.uint8).reshape(num, rows*cols))

    images[0] = np.append(images[0], images[2], axis=1) # shape = (60000, 1568)
    images[1] = np.append(images[1], images[3], axis=1) # shape = (10000, 1568)

    return ((images[0], labels[0]), (images[1], labels[1]))


# import gzip
# import numpy as np
# import pickle
# import os.path
# from scipy.ndimage.filters import gaussian_filter

# def read_mnist():
#     mnist_dog_file = 'data/mnist_dog.pickle'

#     if os.path.isfile(mnist_dog_file):
#         print('Reading {}...'.format(mnist_dog_file))
#         with open(mnist_dog_file, 'rb') as f:
#             train_set, test_set = pickle.load(f)
#     else:
#         print('Cannot find ' + mnist_dog_file)
#         print('Reading original MNIST...')
#         with gzip.open('data/mnist.pkl.gz', 'rb') as f:
#             train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
#         train_set = (np.append(train_set[0], valid_set[0], axis=0), np.append(train_set[1], valid_set[1], axis=0))

#         print('Preprocessing with DoG filter...')
#         new_train_set = (np.empty((60000, 784 * 2), dtype=np.float32), train_set[1])
#         new_test_set = (np.empty((10000, 784 * 2), dtype=np.float32), test_set[1])
#         for i in range(2):
#             data_set = (train_set, test_set)[i]
#             new_data_set = (new_train_set, new_test_set)[i]

#             for j in range(data_set[1].size):
#                 im1 = gaussian_filter(data_set[0][j].reshape((28,28)), sigma=1)
#                 im2 = gaussian_filter(data_set[0][j].reshape((28,28)), sigma=2)

#                 im_on = im1 - im2
#                 im_on[im_on < 0] = 0
#                 im_on /= im_on.max()
#                 im_on = im_on.reshape((784,))

#                 im_off = im2 - im1
#                 im_off[im_off < 0] = 0
#                 im_off /= im_off.max()
#                 im_off = im_off.reshape((784,))

#                 new_data_set[0][j] = np.append(im_on, im_off)
#         train_set = new_train_set
#         test_set = new_test_set

#         with open(mnist_dog_file, 'wb') as f:
#             pickle.dump((train_set, test_set), f)

#     return train_set, test_set
