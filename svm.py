#!/usr/bin/env python

from sklearn import svm
from readers.tidigits import read_data

n_bands=40
n_frames=41

train_set, test_set = read_data(path='data/tidigits', n_bands=n_bands, n_frames=n_frames)

clf = svm.SVC(kernel='linear')
clf.fit(train_set[0].reshape((-1, n_bands * n_frames)), train_set[1])
accuracy = clf.score(test_set[0].reshape((-1, n_bands * n_frames)), test_set[1])
print('Accuracy:', accuracy * 100)
