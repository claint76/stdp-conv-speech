#!/usr/bin/env python3

from python_speech_features import fbank
import numpy as np
import scipy.io.wavfile as wav
from sklearn.preprocessing import normalize
import os


def read_data(path, n_bands, n_frames):
    overlap = 0.5

    # tidigits_file = 'data/tidigits/tidigits_{}_{}.pickle'.format(n_bands, n_frames)
    # if os.path.isfile(tidigits_file):
    #     print('Reading {}...'.format(tidigits_file))
    #     with open(tidigits_file, 'rb') as f:
    #         train_set, test_set = pickle.load(f)
    #     return train_set, test_set

    # wordlist = ['that', 'she', 'an', 'all', 'your', 'me', 'had', 'like', 'don\'t', 'and', 'year', 'water', 'dark', 'of', 'rag', 'oily', 'wash', 'ask', 'carry', 'suit']
    # wordlist = ['that', 'she', 'an', 'all', 'your', 'me', 'had', 'like', 'don\'t', 'and']
    wordlist = ['that', 'she', 'all', 'your', 'me', 'had', 'like', 'don\'t', 'year', 'water', 'dark', 'rag', 'oily', 'wash', 'ask', 'carry', 'suit']

    labels = []
    feats = []

    filelist = []
    for root, dirs, files in os.walk(path):
        paths = (os.path.join(root, file) for file in files if file.endswith('.wrd'))
        for path in paths:
            f = open(path)
            for line in f.readlines():
                start, end, word = line.split()
                start, end = int(start), int(end)
                if word not in wordlist:
                    continue
                if start >= end: # train/{dr5/mmcc0/sx348.wrd, dr7/fmkc0/sx352.wrd}
                    # print('{}: {} {} {}'.format(path, start, end, word))
                    continue

                rate, sig = wav.read(path.rpartition('.')[0] + '.new.wav') # .new.wav files are generated using sox
                sig = sig[start:end]
                duration = sig.size / rate

                winlen = duration / (n_frames * (1 - overlap) + overlap)
                winstep = winlen * (1 - overlap)
                feat, energy = fbank(sig, rate, winlen, winstep, nfilt=n_bands, nfft=4096, winfunc=np.hamming)
                feat = np.log(feat)

                if feat.shape[0] < n_frames:
                    continue

                feats.append(feat[:n_frames].flatten())  # feat may have 41 or 42 frames
                labels.append(wordlist.index(word))
            f.close()

    feats = np.stack(feats)
    labels = np.array(labels, dtype=np.uint8)
    feats = normalize(feats, norm='l2', axis=1)

    np.random.seed(42)
    p = np.random.permutation(len(labels))
    feats, labels = feats[p], labels[p]

    n_train_samples = int(len(labels) * 0.7)

    train_set = (feats[:n_train_samples], labels[:n_train_samples])
    test_set = (feats[n_train_samples:], labels[n_train_samples:])

    # with open(tidigits_file, 'wb') as f:
    #     pickle.dump((train_set, test_set), f)

    return train_set, test_set
