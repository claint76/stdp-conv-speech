#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle


with open('../params_globalpool.json') as f:
    params = json.load(f)


rows = 5
cols = params['layers'][1]['map_num'] // rows

with open('weights_layer_1.pickle', 'rb') as f:
    w = pickle.load(f)
w = w.reshape((params['layers'][1]['map_num'], params['layers'][0]['map_num'], params['layers'][1]['win'], params['layers'][1]['win']))
w = w.transpose((0, 2, 3, 1))
w = np.insert(w, 2, 0., axis=3)

for i in range(params['layers'][1]['map_num']):
    plt.subplot(rows, cols, i+1)
    plt.axis('off')
    plt.imshow(w[i], interpolation="nearest", vmin=0, vmax=1)

plt.show()
