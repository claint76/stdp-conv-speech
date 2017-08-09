#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle


with open('../params_globalpool.json') as data_file:
    params = json.load(data_file)

# conv1
with open('weights_layer_1.pickle', 'rb') as f:
    w_pre = pickle.load(f)
w_pre = w_pre.reshape((params['layers'][1]['map_num'], params['layers'][0]['map_num'], params['layers'][1]['win'], params['layers'][1]['win']))
w_pre = w_pre.transpose((1, 0, 2, 3))

# conv2
with open('weights_layer_3.pickle', 'rb') as f:
    w = pickle.load(f)
w = w.reshape((params['layers'][3]['map_num'], params['layers'][1]['map_num'], params['layers'][3]['win'], params['layers'][3]['win']))

for i in range(100):
    im0 = np.empty([30, w[0][0].shape[0] * w_pre[0][0].shape[0], w[0][0].shape[1] * w_pre[0][0].shape[1]])
    for j in range(30):
        im0[j] = np.bmat([[w_pre[0][j] * w[i][j][k][l] for l in range(w[0][0].shape[1])] for k in range(w[0][0].shape[0])])
    im0 = np.max(im0, axis=0)

    im1 = np.empty([30, w[0][0].shape[0] * w_pre[0][0].shape[0], w[0][0].shape[1] * w_pre[0][0].shape[1]])
    for j in range(30):
        im1[j] = np.bmat([[w_pre[1][j] * w[i][j][k][l] for l in range(w[1][0].shape[1])] for k in range(w[1][0].shape[0])])
    im1 = np.max(im1, axis=0)

    im = np.stack([im0, im1, np.zeros_like(im0)], axis=-1)

    plt.subplot(10, 10, i+1)
    plt.axis('off')
    plt.imshow(im, interpolation="nearest", vmin=0, vmax=1)

plt.show()
