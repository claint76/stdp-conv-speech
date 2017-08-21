#!/usr/bin/env python3

import argparse
import json
import numpy as np
import pickle
import sys
import time
from sklearn import svm

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from network import Network
from readers.mnist import read_data


parser = argparse.ArgumentParser()
parser.add_argument('params_file')
parser.add_argument('-s', '--stage', choices=['train', 'test'])
parser.add_argument('-l', '--train_from_layer', type=int)
args = parser.parse_args()


to_train = True
to_test = True

if args.stage:
    if args.stage == 'train':
        to_test = False
    elif args.stage == 'test':
        to_train = False

if args.train_from_layer:
    train_from_layer = args.train_from_layer
else:
    train_from_layer = 1


weights_path = 'output/weights_layer_{}.pickle'


print('Reading data...')
train_set, test_set = read_data()


print('Creating network...')
with open(args.params_file) as f:
    params = json.load(f)
network = Network(params)


record = None
if params['layers'][-1]['type'] == 'supe':
    record = 'spike_time'
elif params['layers'][-1]['type'] == 'globalpool':
    record = 'V'


def print_progress(progress):
    print('\r[{:50}] {:6.2f}%'.format('#' * int(progress * 50), progress * 100), end='', flush=True)
    if progress == 1:
        print()


def run(data_set, output=None, learning_round=None):
    for k in range(data_set[1].size):
        network.reset()
        with np.errstate(divide='ignore'):
            network.layers[0].spike_time.set((30 / data_set[0][k]).astype(np.float32))
        if hasattr(network.active_layers[-1], 'label'):
            network.active_layers[-1].label.fill(data_set[1][k])

        for j in range(20):
            network.step()

            if output is not None and record == 'spike_time':
                spikes = network.layers[-1].spikes.get()[:network.layers[-1].spike_count.get()[0]]
                output[0][k][spikes] = network.it

        if output is not None and record == 'V':
            network.layers[-1].V.get(output[0][k])
            output[1][k] = data_set[1][k]

        print_progress((k + 1) / data_set[1].size)


if to_train:
    print('Training...')
    start_time = time.time()

    for i, layer in enumerate(network.layers):
        network.active_layers.append(layer)
        if hasattr(layer, 'plastic'):
            if i < train_from_layer: # train from layer x
                with open(weights_path.format(i), 'rb') as f:
                    layer.weights.set(pickle.load(f))
                    continue

            print('Training layer {} for {} rounds...'.format(i, layer.learning_rounds))
            layer.plastic.fill(True)
            for r in range(layer.learning_rounds):
                run(train_set, learning_round=r)
                with open(weights_path.format(str(i)+'_'+str(r)), 'wb') as f:
                    pickle.dump(layer.weights.get(), f)

            layer.plastic.fill(False)

            with open(weights_path.format(i), 'wb') as f:
                pickle.dump(layer.weights.get(), f)

    print('Training time: {:.2f} seconds'.format(time.time() - start_time))


if to_test:
    print('Testing...')
    start_time = time.time()

    if not to_train:
        for i, layer in enumerate(network.layers):
            if hasattr(layer, 'weights'):
                with open(weights_path.format(i), 'rb') as f:
                    layer.weights.set(pickle.load(f))

    network.active_layers = network.layers

    def get_output(n):
        output = (np.empty((n, network.layers[-1].layer_size), dtype=np.float32), np.empty((n,), dtype=np.int8))
        if record == 'spike_time':
            output[0].fill(np.inf)
        return output

    print('Testing on train_set...')
    train_output = get_output(train_set[1].size)
    run(train_set, train_output)
    with open('output/output_train_set.pickle', 'wb') as f:
        pickle.dump(train_output, f)

    print('Testing on test_set...')
    test_output = get_output(test_set[1].size)
    run(test_set, test_output)
    with open('output/output_test_set.pickle', 'wb') as f:
        pickle.dump(test_output, f)

    print('Testing time: {:.2f} seconds'.format(time.time() - start_time))


    if params['layers'][-1]['type'] == 'globalpool':
        print('Running SVM on V of global pooling layer...')
        start_time = time.time()

        clf = svm.SVC(kernel='linear')
        clf.fit(train_output[0], train_output[1])
        accuracy = clf.score(test_output[0], test_output[1])
        print('Accuracy:', accuracy * 100)

        print('SVM time: {:.2f} seconds'.format(time.time() - start_time))
    elif params['layers'][-1]['type'] == 'supe':
        print('Calculating accuracy of first spikes of supervised layer...')

        accuracy = (np.argmin(test_output[0], axis=1) == test_output[1]).sum() / test_set[1].size
        print('Accuracy:', accuracy * 100)
