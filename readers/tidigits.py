#!/usr/bin/env python3

from python_speech_features import fbank
import numpy as np
import scipy.io.wavfile as wav
import os
import pickle
import matplotlib.pyplot as plt


def read_data():
    n_frames = 40
    overlap = 0.5

    tidigits_file = 'data/tidigits/tidigits.pickle'

    if os.path.isfile(tidigits_file):
        print('Reading {}...'.format(tidigits_file))
        with open(tidigits_file, 'rb') as f:
            train_set, test_set = pickle.load(f)
    else:
        filelist = []
        for root, dirs, files in os.walk('data/tidigits/MAN'):
            for file in files:
                if file.endswith('.waV') and file[0] != 'O':
                    filelist.append(os.path.join(root, file))
        n_samples = len(filelist)

        def keyfunc(x):
            s = x.split('/')
            return (s[-1][0], s[-2], s[-1][1]) # BH/1A_endpt.wav: sort by '1', 'BH', 'A'
        filelist.sort(key=keyfunc)

        feats = np.empty((n_samples, 26 * n_frames))
        labels = np.empty((n_samples,), dtype=np.uint8)

        for i, file in enumerate(filelist):
            label = file.split('/')[-1][0]
            if label == 'Z':
                labels[i] = np.uint8(0)
            else:
                labels[i] = np.uint8(label)

            rate, sig = wav.read(file)
            duration = sig.size / rate
            winlen = duration / (n_frames * (1 - overlap) + overlap)
            winstep = winlen * (1 - overlap)
            feat, energy = fbank(sig, rate, winlen, winstep, nfft=4096, winfunc=np.hamming)
            feat = np.log(feat)

            feats[i] = feat[:n_frames].flatten() # feat may have 40 or 41 frames

        p = np.random.permutation(n_samples)
        feats, labels = feats[p], labels[p]

        train_set = (feats[:n_samples//2], labels[:n_samples//2])
        test_set = (feats[n_samples//2:], labels[n_samples//2:])

        with open(tidigits_file, 'wb') as f:
            pickle.dump((train_set, test_set), f)

    return train_set, test_set
