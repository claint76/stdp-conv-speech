#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle


with open('../params_globalpool.json') as f:
    params = json.load(f)


rows = 10
cols = params['layers'][1]['map_num'] // rows

with open('weights_layer_1.pickle', 'rb') as f:
    w = pickle.load(f)
w = w.reshape((params['layers'][1]['sec_num'], params['layers'][1]['map_num'], params['layers'][1]['win'][0], params['layers'][1]['win'][1]))

for i in range(params['layers'][1]['sec_num']):
    plt.figure()
    for j in range(params['layers'][1]['map_num']):
        plt.subplot(rows, cols, j+1)
        plt.axis('off')
        plt.imshow(w[i][j], interpolation="nearest", vmin=0, vmax=1)

plt.show()
