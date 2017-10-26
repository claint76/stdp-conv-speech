#!/usr/bin/env python3

import json
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append('..')
from readers.tidigits import read_data


with open('../params_globalpool.json') as f:
    params = json.load(f)
    params['layers'][0]['height'] = params['layers'][1]['sec_num'] * params['layers'][1]['sec_size'] + params['layers'][1]['win'][0] - 1


print('Reading data...')
train_set, test_set = read_data(path='../data/tidigits', n_bands=params['layers'][0]['width'], n_frames=params['layers'][0]['height'])

d = train_set[0][0]
n = np.count_nonzero(d)
indices = np.flip(np.argsort(d), axis=0)
spikes_per_packet = 20
packet_count = (n + spikes_per_packet - 1) // spikes_per_packet

t = d.astype(np.float32)
t[indices[:n]] = np.repeat(np.arange(packet_count), spikes_per_packet)[:n]
t[indices[n:]] = np.inf


d = d.reshape((41, 40))
d = d.transpose()
t = t.reshape((41, 40))
t = t.transpose()

# plotting
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5.2, 3.2))

axes[0].imshow(d, origin='lower')
axes[0].add_patch(
    patches.Rectangle(
        (12.6, -0.4),
        1,
        40,
        fill=False
    )
)
axes[0].set_ylabel('Frequency bands')
axes[0].set_xlabel('Time frames')

axes[1].scatter(t[:,13], np.arange(40), s=10)
axes[1].set_ylabel('Frequency bands')
axes[1].set_xlabel('Time frames')

fig.tight_layout()
fig.savefig('input_coding.tiff', dpi=300)
plt.show()
