#!/usr/bin/env python3

import json
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.manifold import TSNE

sys.path.append('..')
from readers.tidigits import read_data


with open('../params.json') as f:
    params = json.load(f)
    params['layers'][0]['height'] = params['layers'][1]['sec_num'] * params['layers'][1]['sec_size'] + params['layers'][1]['win'][0] - 1

n_bands = params['layers'][0]['width']
n_frames = params['layers'][0]['height']

print('Reading data...')
train_set, test_set = read_data(path='../data/tidigits', n_bands=n_bands, n_frames=n_frames)

with open('../output/output_test_set.pickle', 'rb') as f:
    output_test_set = pickle.load(f)


fig, axes = plt.subplots(1, 2, figsize=(5.2, 4))

X_tsne = TSNE().fit_transform(test_set[0])
axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=test_set[1], cmap='tab10')
axes[0].axis('off')
axes[0].set_title('(A) t-SNE of original data')

X_tsne = TSNE().fit_transform(output_test_set[0])
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=output_test_set[1], cmap='tab10')
axes[1].axis('off')
axes[1].set_title('(B) t-SNE of SNN output')

fig.savefig('t-sne.png')
plt.show()