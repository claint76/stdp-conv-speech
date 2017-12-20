#!/usr/bin/env python3

import json
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append('..')
from readers.tidigits import read_data


with open('../params.json') as f:
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
t = t.reshape((n_frames, n_bands))

# plotting
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.9))

axes[0].imshow(d, extent=(0, d.shape[1], d.shape[0], 0))
axes[0].add_patch(patches.Rectangle((0, 17), n_bands, 1, fill=False))
axes[0].set_xlabel('Frequency bands')
axes[0].set_ylabel('Frames')
# axes[0].text(-0.1, 1.1, 'A', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes, size=14, weight='bold')

axes[1].scatter(t[18, :], np.arange(n_bands), s=5)
axes[1].set_xlabel('Time steps')
axes[1].set_ylabel('Frequency bands')
# axes[1].text(-0.1, 1.1, 'B', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes, size=14, weight='bold')

con1 = patches.ConnectionPatch(xyA=(0, 0), xyB=(40, 18), coordsA='axes fraction', coordsB='data', axesA=axes[1], axesB=axes[0])
con2 = patches.ConnectionPatch(xyA=(0, 1), xyB=(40, 17), coordsA='axes fraction', coordsB='data', axesA=axes[1], axesB=axes[0])
axes[1].add_artist(con1)
axes[1].add_artist(con2)

fig.tight_layout()
fig.subplots_adjust(top=0.85, wspace=0.3)
fig.savefig('input_coding.png')
plt.show()
