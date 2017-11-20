#!/usr/bin/env python

import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('../output/output_test_set.pickle', 'rb') as f:
    test_set = pickle.load(f)

fig, axes = plt.subplots(5, 2, figsize=(4.5, 4))

for i in range(10):
    idxes = np.where(test_set[1] == i)[0]
    idx = idxes[0]
    ax = axes.ravel()[i]
    ax.imshow(test_set[0][idx].reshape((40, 9)).transpose())
    ax.axis('off')
    ax.set_title('Digit ' + str(i))

fig.tight_layout()
fig.savefig('pooling.png')
plt.show()
