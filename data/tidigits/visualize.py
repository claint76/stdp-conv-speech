#!/usr/bin/env python3

from python_speech_features import fbank
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct, idct
from sklearn.preprocessing import normalize
import os
import pickle
import matplotlib.pyplot as plt


n_bands = 41
n_frames = 40
overlap = 0.5

filelist = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.waV') and file[0] != 'O':
            filelist.append(os.path.join(root, file))
n_samples = len(filelist)

def keyfunc(x):
    s = x.split('/')
    return (s[-1][0], s[-2], s[-1][1]) # BH/1A_endpt.wav: sort by '1', 'BH', 'A'
filelist.sort(key=keyfunc)

for i, file in enumerate(filelist):

    rate, sig = wav.read(file)
    duration = sig.size / rate
    winlen = duration / (n_frames * (1 - overlap) + overlap)
    winstep = winlen * (1 - overlap)
    feat, energy = fbank(sig, rate, winlen, winstep, nfilt=n_bands, nfft=4096, winfunc=np.hamming)
    feat = np.log(feat)
    plt.subplot(131)
    plt.imshow(feat)

    feat2 = feat.copy()
    feat2[feat2 < 4] = 0
    plt.subplot(132)
    plt.imshow(feat2)

    feat3 = dct(feat, type=2, axis=1, norm='ortho')#[:,:n_bands//2]
    feat3[:,n_bands//2:] = 0
    feat3 = idct(feat3, type=2, axis=1, norm='ortho')
    plt.subplot(133)
    plt.imshow(feat3)

    plt.show()
