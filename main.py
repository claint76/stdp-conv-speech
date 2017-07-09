#!/usr/bin/env python

import json
import numpy as np
import pickle
import sys
import time

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from mnist import read_mnist
from network import Network


print('Reading MNIST...')
train_set, test_set = read_mnist()


print('Creating network...')
with open('params.json') as f:
    params = json.load(f)
network = Network(params)


to_train = True
to_test = True

if len(sys.argv) > 1:
    if sys.argv[1] == 'train':
        to_test = False
    elif sys.argv[1] == 'test':
        to_train = False
    else:
        print('Invalid argument!')
        exit(1)

if to_train:
    train_from_layer = 1

weights_path = 'output/weights_layer_{}.pickle'


record = None
if params['layers'][-1]['type'] == 'supe':
    record = 'spike_time'
elif params['layers'][-1]['type'] == 'globalpool':
    record = 'V'


def print_progress(progress):
    print('\r[{:50}] {:6.2f}%'.format('#' * int(progress * 50), progress * 100), end='', flush=True)
    if progress == 1:
        print()


def run(data_set, output=None):
    for i in range(data_set[1].size):
        network.reset()
        with np.errstate(divide='ignore'):
            network.layers[0].spike_time.set(30 / data_set[0][i].astype(np.float32))
        if hasattr(network.active_layers[-1], 'label'):
            network.active_layers[-1].label.fill(data_set[1][i])

        for j in range(10):
            network.step()

            if output is not None and record == 'spike_time':
                spikes = network.layers[-1].spikes.get()[:network.layers[-1].spike_count.get()[0]]
                output[0][i][spikes] = network.it

        if output is not None and record == 'V':
            network.layers[-1].V.get(output[0][i])
            output[1][i] = data_set[1][i]

        print_progress((i + 1) / data_set[1].size)


if to_train:
    print('Training...')
    start_time = time.time()

    for layer in network.layers:
        network.active_layers.append(layer)
        if hasattr(layer, 'plastic'):
            i = network.layers.index(layer)

            if i < train_from_layer: # train from layer x
                with open(weights_path.format(i), 'rb') as f:
                    layer.weights.set(pickle.load(f))
                    continue

            print('Training layer {} for {} rounds...'.format(i, layer.learning_rounds))
            layer.plastic.fill(True)
            for r in range(layer.learning_rounds):
                run(train_set)
            layer.plastic.fill(False)

            with open(weights_path.format(i), 'wb') as f:
                pickle.dump(layer.weights.get(), f)

    print('Training time: {} seconds'.format(time.time() - start_time))


if to_test:
    print('Testing...')
    start_time = time.time()

    if not to_train:
        for layer in network.layers:
            if hasattr(layer, 'weights'):
                with open(weights_path.format(network.layers.index(layer)), 'rb') as f:
                    layer.weights.set(pickle.load(f))

    network.active_layers = network.layers

    def get_output(n):
        output = (np.empty((n, network.layers[-1].layer_size), dtype=np.float32), np.empty((n,), dtype=np.int8))
        if record == 'spike_time':
            output[0].fill(np.inf)
        return output

    print('Testing on train_set...')
    output = get_output(train_set[1].size)
    run(train_set, output)
    with open('output/output_train_set.pickle', 'wb') as f:
        pickle.dump(output, f)

    print('Testing on test_set...')
    output = get_output(test_set[1].size)
    run(test_set, output)
    with open('output/output_test_set.pickle', 'wb') as f:
        pickle.dump(output, f)

    print('Testing time: {} seconds'.format(time.time() - start_time))
