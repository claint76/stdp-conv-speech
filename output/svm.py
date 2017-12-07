#!/usr/bin/env python

from sklearn import svm
import pickle


with open('output_train_set.pickle', 'rb') as f:
    train_set = pickle.load(f)
with open('output_test_set.pickle', 'rb') as f:
    test_set = pickle.load(f)

clf = svm.SVC(kernel='linear')
clf.fit(train_set[0], train_set[1])
accuracy = clf.score(test_set[0], test_set[1])
print('Accuracy:', accuracy * 100)
