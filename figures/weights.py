#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle


with open('../params_globalpool.json') as f:
    params = json.load(f)


rows = 4
cols = params['layers'][1]['map_num'] // rows

with open('../output/weights_layer_1.pickle', 'rb') as f:
    w = pickle.load(f)
w = w.reshape((params['layers'][1]['sec_num'], params['layers'][1]['map_num'], params['layers'][1]['win'][0], params['layers'][1]['win'][1]))

plt.figure(figsize=(6.5, 8))
for j in range(params['layers'][1]['map_num']):
    plt.subplot(rows, cols, j+1)
    plt.axis('off')
    plt.imshow(w[5][j].transpose(), interpolation="nearest", vmin=0, vmax=1)

plt.savefig('weights.png')
plt.show()
