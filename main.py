import json
import numpy as np
import pickle

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from mnist import read_mnist
from network import Network


train_set, test_set = read_mnist()

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
