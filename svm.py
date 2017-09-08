#!/usr/bin/env python3

from sklearn import svm
from readers.tidigits import read_data

train_set, test_set = read_data()

clf = svm.SVC(kernel='linear')
clf.fit(train_set[0].reshape((932, 40*26)), train_set[1])
accuracy = clf.score(test_set[0].reshape((932, 40*26)), test_set[1])
print('Accuracy:', accuracy * 100)
