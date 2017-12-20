#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import json
import pickle


with open('../params.json') as f:
    params = json.load(f)

m = 3  # features
n = 12 # time points
fig, axes = plt.subplots(n, m, figsize=(4.5, 4.0))

for i in range(n):
    with open('../output/weights_layer_1_{}.pickle'.format(i * 2000), 'rb') as f:
        w = pickle.load(f)
    w = w.reshape((params['layers'][1]['sec_num'], 40, params['layers'][1]['win'][0], params['layers'][1]['win'][1]))
    axes[i, 0].axis('off')
    axes[i, 0].imshow(w[5][2], interpolation="nearest", vmin=0, vmax=1)
    axes[i, 1].axis('off')
    axes[i, 1].imshow(w[5][6], interpolation="nearest", vmin=0, vmax=1)
    axes[i, 2].axis('off')
    axes[i, 2].imshow(w[5][34], interpolation="nearest", vmin=0, vmax=1)

fig.subplots_adjust(left=0.06, right=0.94, bottom=0.05, top=0.95, wspace=0.4, hspace=0.01)
fig.savefig('weights_changing.png')
img = image.imread('weights_changing.png')

fig, axes = plt.subplots(figsize=(5.2, 4.5))
axes.imshow(img)
axes.set_yticks([y for y in range(101, 1100, 90)])
axes.set_yticklabels([y*2000 for y in range(12)])
axes.set_xticks([250, 690, 1130])
axes.set_xticklabels(['Feature map #1', 'Feature map #2', 'Feature map #3'])
axes.set_ylabel('Number of training samples')

fig.savefig('weights_changing.png')
plt.show()
