#!/usr/bin/env python3

from sklearn import svm
from readers.tidigits import read_data

n_bands = 41
n_frames = 40

train_set, test_set = read_data(n_bands, n_frames)

clf = svm.SVC(kernel='linear')
clf.fit(train_set[0].reshape((-1, n_bands * n_frames)), train_set[1])
accuracy = clf.score(test_set[0].reshape((-1, n_bands * n_frames)), test_set[1])
print('Accuracy:', accuracy * 100)
