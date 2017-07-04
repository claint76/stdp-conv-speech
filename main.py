#!/usr/bin/env python

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

def print_progress(progress):
    print("\r[{:50}] {:5.2f}%".format('#' * int(progress * 50), progress * 100), end="", flush=True)
    if progress == 1:
        print()


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
            if output is not None:
                network.layers[-1].V.get(output[i])
            print_progress((i + 1) / data_set[1].size)

    if (phase == 0):
        for l, layer in enumerate(network.layers):
            if hasattr(layer, 'plastic'):
                layer.plastic.fill(True)
                for r in range(layer.learning_rounds):
                    run()
                layer.plastic.fill(False)
                with open('output/weights_layer_{}.pickle'.format(l), 'wb') as f:
                    pickle.dump(layer.weights.get(), f)
    else:
        output = np.empty((data_set[1].size, network.layers[-1].layer_size), dtype=np.float32)
        run(output)

cuda.stop_profiler()
