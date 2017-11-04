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

n_bands = params['layers'][0]['width']
n_frames = params['layers'][0]['height']

print('Reading data...')
train_set, test_set = read_data(path='../data/tidigits', n_bands=n_bands, n_frames=n_frames)

d = train_set[0][0]
n = np.count_nonzero(d)
indices = np.flip(np.argsort(d), axis=0)
spikes_per_packet = 20
packet_count = (n + spikes_per_packet - 1) // spikes_per_packet

t = d.astype(np.float32)
t[indices[:n]] = np.repeat(np.arange(packet_count), spikes_per_packet)[:n]
t[indices[n:]] = np.inf


d = d.reshape((n_frames, n_bands))
d = d.transpose()
t = t.reshape((n_frames, n_bands))
t = t.transpose()

# plotting
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5.2, 2.69))

axes[0].imshow(d, origin='lower')
axes[0].add_patch(
    patches.Rectangle(
        (17.5, -0.5),
        1,
        n_bands,
        fill=False
    )
)
axes[0].set_ylabel('Frequency bands')
axes[0].set_xlabel('Time frames')
axes[0].set_ylim(-0.5, n_bands-1+0.5)
axes[0].set_xlim(-0.5, n_frames-1+0.5)

axes[1].scatter(t[:,18], np.arange(n_bands), s=5)
axes[1].set_ylabel('Frequency bands')
axes[1].set_xlabel('Time frames')

fig.tight_layout()
fig.savefig('input_coding.png')
plt.show()
