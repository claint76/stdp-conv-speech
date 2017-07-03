import gzip
import json
import numpy
import pickle

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from network import Network


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

with open('params.json') as f:    
    params = json.load(f)
network = Network(params)


for phase in range(1): # 0: train on train_set, 1: test on train_set, 2: test on test_set
    print("Phase {}:".format(phase))

    print("Reading MNIST DoG version...")
    if (phase == 2):
        data_set = test_set
    else:
        data_set = train_set

    print("Simulating...")
    # for i in range(data_set[1].size):
    for i in range(1000):
        print('i = ', i)
        network.reset()

        network.layers[0].spike_time.set(1 - data_set[0][i])

        for j in range(10):
            network.step()
            #inhibition

