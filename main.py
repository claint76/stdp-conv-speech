import gzip
import json
import numpy as np
import pickle
from scipy.ndimage.filters import gaussian_filter

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from network import Network


print('Reading MNIST...')
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

print('Preprocessing with DoG filter...')
new_train_set = (np.empty((50000, 784 * 2), dtype=np.float32), train_set[1])
new_test_set = (np.empty((50000, 784 * 2), dtype=np.float32), test_set[1])
for i in range(2):
    data_set = (train_set, test_set)[i]
    new_data_set = (new_train_set, new_test_set)[i]

    for j in range(data_set[1].size):
        im1 = gaussian_filter(data_set[0][j].reshape((28,28)), sigma=1)
        im2 = gaussian_filter(data_set[0][j].reshape((28,28)), sigma=2)

        im_on = im1 - im2
        im_on[im_on < 0] = 0
        im_on /= im_on.max()
        im_on = im_on.reshape((784,))

        im_off = im2 - im1
        im_off[im_off < 0] = 0
        im_off /= im_off.max()
        im_off = im_off.reshape((784,))

        new_data_set[0][j] = np.append(im_on, im_off)
train_set = new_train_set
test_set = new_test_set


with open('params.json') as f:    
    params = json.load(f)
network = Network(params)


for phase in range(3): # 0: train on train_set, 1: test on train_set, 2: test on test_set
    print("Phase {}:".format(phase))

    if (phase == 2):
        data_set = test_set
    else:
        data_set = train_set

    print("Simulating...")
    def run(output=None):
        for i in range(data_set[1].size):
            network.reset()
            network.layers[0].spike_time.set(1 - data_set[0][i])
            for j in range(10):
                network.step()
                network.inhibit()
            if output:
                network.layers[-1].V.get(output[i])

    if (phase == 0):
        for l, layer in enumerate(network.layers):
            if hasattr(layer, 'plastic'):
                layer.plastic.fill(True)
                for r in range(layer.learning_rounds):
                    run()
                layer.plastic.fill(False)
                with open('weights_layer_{}.pickle'.format(l), 'wb') as f:
                    pickle.dump(layer.weights.get(), f)
    else:
        ouput = np.empty((data_set[1].size, network.layers[-1].layer_size), dtype=np.float32)
        run(output)
