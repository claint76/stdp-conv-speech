#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import pdb


plt.ion()

with open('../params_globalpool.json') as f:
    params = json.load(f)

rows = 5
cols = params['layers'][1]['map_num'] // rows

ims = []
for i in range(params['layers'][1]['map_num']):
    plt.subplot(rows, cols, i+1)
    plt.axis('off')
    ims.append(plt.imshow(np.zeros((params['layers'][1]['win'], params['layers'][1]['win'], 3)), interpolation="nearest", vmin=0, vmax=1))
plt.pause(0.1)

for k in range(0, 60000, 100):
    with open('a/w_{}.pickle'.format(k), 'rb') as f:
        w = pickle.load(f)
    w = w.reshape((params['layers'][1]['map_num'], params['layers'][0]['map_num'], params['layers'][1]['win'], params['layers'][1]['win']))
    w = w.transpose((0, 2, 3, 1))
    w = np.insert(w, 2, 0., axis=3)

    for i in range(params['layers'][1]['map_num']):
        ims[i].set_data(w[i])
    plt.draw()
    plt.pause(0.1)
